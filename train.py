import shutil
from pathlib import Path

import tensorflow as tf

import sle_gan

args = sle_gan.get_args()
print(args)

# For debugging:
# tf.config.experimental_run_functions_eagerly(True)

physical_devices = tf.config.list_physical_devices("GPU")
_ = [tf.config.experimental.set_memory_growth(x, True) for x in physical_devices]

experiments_folder = Path("logs") / args.name
if experiments_folder.is_dir():
    if args.override:
        shutil.rmtree(experiments_folder)
    else:
        raise FileExistsError("Experiment already exists")
checkpoints_folder = experiments_folder / "checkpoints"
checkpoints_folder.mkdir(parents=True)
logs_folder = experiments_folder / "tensorboard_logs"
logs_folder.mkdir(parents=True)

RESOLUTION = args.resolution
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
DATA_FOLDER = args.data_folder

dataset = sle_gan.create_dataset(batch_size=BATCH_SIZE, folder=DATA_FOLDER, resolution=RESOLUTION,
                                 use_flip_augmentation=True, shuffle_buffer_size=500)

G = sle_gan.Generator(RESOLUTION)
sample_G_output = G.initialize(BATCH_SIZE)
if args.generator_weights is not None:
    G.load_weights(args.generator_weights)
    print("Weights are loaded for G")
print(f"[Model G] output shape: {sample_G_output.shape}")

D = sle_gan.Discriminator(RESOLUTION)
sample_D_output = D.initialize(BATCH_SIZE)
if args.discriminator_weights is not None:
    D.load_weights(str(checkpoints_folder / "D_checkpoint.h5"))
    print("Weights are loaded for D")
print(f"[Model D] real_fake output shape: {sample_D_output[0].shape}")
print(f"[Model D] image output shape{sample_D_output[1].shape}")
print(f"[Model D] image part output shape{sample_D_output[2].shape}")

G_optimizer = tf.optimizers.Adam(learning_rate=args.G_learning_rate)
D_optimizer = tf.optimizers.Adam(learning_rate=args.D_learning_rate)

if args.fid:
    # Model for the FID calculation
    fid_inception_model = sle_gan.InceptionModel(height=RESOLUTION, width=RESOLUTION)

test_input_size = 25
test_input_for_generation = sle_gan.create_input_noise(test_input_size)
test_images = sle_gan.get_test_images(test_input_size, DATA_FOLDER, RESOLUTION)

tb_file_writer = tf.summary.create_file_writer(str(logs_folder))
tb_file_writer.set_as_default()

G_loss_metric = tf.keras.metrics.Mean()
D_loss_metric = tf.keras.metrics.Mean()
D_real_fake_loss_metric = tf.keras.metrics.Mean()
D_I_reconstruction_loss_metric = tf.keras.metrics.Mean()
D_I_part_reconstruction_loss_metric = tf.keras.metrics.Mean()

diff_augment_policies = None
if args.diff_augment:
    diff_augment_policies = "color,translation,cutout"

for epoch in range(EPOCHS):
    print(f"Epoch {epoch} -------------")
    for step, image_batch in enumerate(dataset):
        G_loss, D_loss, D_real_fake_loss, D_I_reconstruction_loss, D_I_part_reconstruction_loss = sle_gan.train_step(
            G=G,
            D=D,
            G_optimizer=G_optimizer,
            D_optimizer=D_optimizer,
            images=image_batch,
            diff_augmenter_policies=diff_augment_policies)

        G_loss_metric(G_loss)
        D_loss_metric(D_loss)
        D_real_fake_loss_metric(D_real_fake_loss)
        D_I_reconstruction_loss_metric(D_I_reconstruction_loss)
        D_I_part_reconstruction_loss_metric(D_I_part_reconstruction_loss)

        if step % 100 == 0 and step != 0:
            print(f"\tStep {step} - "
                  f"G loss {G_loss_metric.result():.4f} | "
                  f"D loss {D_loss_metric.result():.4f} | "
                  f"D realfake loss {D_real_fake_loss_metric.result():.4f} | "
                  f"D I recon loss {D_I_reconstruction_loss_metric.result():.4f} | "
                  f"D I part recon loss {D_I_part_reconstruction_loss_metric.result():.4f}")

    if args.fid:
        if epoch % args.fid_frequency == 0:
            fid_score = sle_gan.evaluation_step(inception_model=fid_inception_model,
                                                dataset=dataset,
                                                G=G,
                                                batch_size=BATCH_SIZE,
                                                image_height=RESOLUTION,
                                                image_width=RESOLUTION,
                                                nb_of_images_to_use=args.fid_number_of_images)
            print(f"[FID] {fid_score:.2f}")
            tf.summary.scalar("FID_score", fid_score, epoch)

    tf.summary.scalar("G_loss/G_loss", G_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_loss", D_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_real_fake_loss", D_real_fake_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_I_reconstruction_loss", D_I_reconstruction_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_I_part_reconstruction_loss", D_I_part_reconstruction_loss_metric.result(), epoch)

    print(f"Epoch {epoch} - "
          f"G loss {G_loss_metric.result():.4f} | "
          f"D loss {D_loss_metric.result():.4f} | "
          f"D realfake loss {D_real_fake_loss_metric.result():.4f} | "
          f"D I recon loss {D_I_reconstruction_loss_metric.result():.4f} | "
          f"D I part recon loss {D_I_part_reconstruction_loss_metric.result():.4f}")

    G_loss_metric.reset_states()
    D_loss_metric.reset_states()
    D_real_fake_loss_metric.reset_states()
    D_I_part_reconstruction_loss_metric.reset_states()
    D_I_reconstruction_loss_metric.reset_states()

    # TODO: save weights only when the FID score gets better
    G.save_weights(str(checkpoints_folder / "G_checkpoint.h5"))
    D.save_weights(str(checkpoints_folder / "D_checkpoint.h5"))

    # Generate test images
    generated_images = G(test_input_for_generation, training=False)
    generated_images = sle_gan.postprocess_images(generated_images, dtype=tf.uint8).numpy()
    sle_gan.visualize_images_on_grid_and_save(epoch, generated_images, experiments_folder / "generated_images",
                                              5, 5)

    # Generate reconstructions from Discriminator
    _, decoded_images, decoded_part_images = D(test_images, training=False)
    decoded_images = sle_gan.postprocess_images(decoded_images, dtype=tf.uint8).numpy()
    decoded_part_images = sle_gan.postprocess_images(decoded_part_images, dtype=tf.uint8).numpy()
    sle_gan.visualize_images_on_grid_and_save(epoch, decoded_images, experiments_folder / "reconstructed_whole_images",
                                              5, 5)
    sle_gan.visualize_images_on_grid_and_save(epoch, decoded_part_images,
                                              experiments_folder / "reconstructed_part_images", 5, 5)
