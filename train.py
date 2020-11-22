import shutil
from pathlib import Path

import tensorflow as tf

import sle_gan

args = sle_gan.get_args()
print(args)

# For debugging:
# tf.config.experimental_run_functions_eagerly(True)

physical_devices = tf.config.list_physical_devices('GPU')
_ = [tf.config.experimental.set_memory_growth(x, True) for x in physical_devices]

experiments_folder = Path("logs") / args.name
if experiments_folder.is_dir():
    if args.override:
        shutil.rmtree(experiments_folder)
    else:
        raise FileExistsError("Experiment already exists")
checkpoints_folder = experiments_folder / "checkpoints"
checkpoints_folder.mkdir(parents=True)
logs_folder = experiments_folder / "logs"
logs_folder.mkdir(parents=True)

RESOLUTION = args.resolution
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.learning_rate
DATA_FOLDER = args.data_folder

dataset = sle_gan.create_dataset(batch_size=BATCH_SIZE, folder=DATA_FOLDER, resolution=RESOLUTION,
                                 use_flip_augmentation=True, shuffle_buffer_size=500)

G = sle_gan.Generator(RESOLUTION)
try:
    G.load_weights(str(checkpoints_folder / "G_checkpoint.h5"))
    print("Weights are loadaed for G")
except:
    pass
sample_G_output = G.initialize(BATCH_SIZE)
print(f"[G] output shape: {sample_G_output.shape}")

D = sle_gan.Discriminator(RESOLUTION)
try:
    D.load_weights(str(checkpoints_folder / "D_checkpoint.h5"))
    print("Weights are loadaed for D")
except:
    pass
sample_D_output = D.initialize(BATCH_SIZE)
print(f"[D] real_fake output shape: {sample_D_output[0].shape}")
print(f"[D] image output shape{sample_D_output[1].shape}")
print(f"[D] image part output shape{sample_D_output[2].shape}")

G_optimizer = tf.optimizers.Adam(learning_rate=LR)
D_optimizer = tf.optimizers.Adam(learning_rate=LR)

test_input_for_generation = sle_gan.create_input_noise(4)
test_images = sle_gan.get_test_images(4, DATA_FOLDER, RESOLUTION)

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

train_step_fn = sle_gan.train_step
if args.v == 2:
    train_step_fn = sle_gan.train_step_v2

for epoch in range(EPOCHS):
    print(f"Epoch {epoch} -------------")
    for step, image_batch in enumerate(dataset):
        G_loss, D_loss, D_real_fake_loss, D_I_reconstruction_loss, D_I_part_reconstruction_loss = train_step_fn(
            G=G,
            D=D,
            G_optimizer=G_optimizer,
            D_optimizer=D_optimizer,
            images=image_batch,
            inject_gaussian_noise=True,
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

    G.save_weights(str(checkpoints_folder / "G_checkpoint.h5"))
    D.save_weights(str(checkpoints_folder / "D_checkpoint.h5"))

    sle_gan.generate_and_save_images(G, epoch, test_input_for_generation, experiments_folder)
    sle_gan.reconstructions(D, epoch, test_images, experiments_folder)
