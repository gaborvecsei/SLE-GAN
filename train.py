import tensorflow as tf

import sle_gan

# For debugging:
# tf.config.experimental_run_functions_eagerly(True)

physical_devices = tf.config.list_physical_devices('GPU')
_ = [tf.config.experimental.set_memory_growth(x, True) for x in physical_devices]

args = sle_gan.get_args()
print(args)

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.learning_rate
DATA_FOLDER = args.data_folder

dataset = sle_gan.create_dataset(batch_size=BATCH_SIZE, folder=DATA_FOLDER, use_flip_augmentation=True,
                                 shuffle_buffer_size=200)

G = sle_gan.Generator()
try:
    G.load_weights("./checkpoints/G_checkpoint.h5")
    print("Weights are loadaed for G")
except:
    pass
sample_G_output = G.initialize()
print(f"[G] output shape: {sample_G_output.shape}")

D = sle_gan.Discriminator()
try:
    sample_D_output = D.initialize()
    print("Weights are loadaed for D")
except:
    pass
print(f"[D] real_fake output shape: {sample_D_output[0].shape}")
print(f"[D] image output shape{sample_D_output[1].shape}")

G_optimizer = tf.optimizers.Adam(learning_rate=LR)
D_optimizer = tf.optimizers.Adam(learning_rate=LR)

test_input_for_generation = sle_gan.create_input_noise(4)

tb_file_writer = tf.summary.create_file_writer("./logs")
tb_file_writer.set_as_default()

G_loss_metric = tf.keras.metrics.Mean()
D_loss_metric = tf.keras.metrics.Mean()
D_real_fake_loss_metric = tf.keras.metrics.Mean()
D_reconstruction_loss_metric = tf.keras.metrics.Mean()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch} -------------")
    for step, image_batch in enumerate(dataset):
        G_loss, D_loss, D_real_fake_loss, D_reconstruction_loss = sle_gan.train_step(G=G,
                                                                                     D=D,
                                                                                     G_optimizer=G_optimizer,
                                                                                     D_optimizer=D_optimizer,
                                                                                     images=image_batch)

        G_loss_metric(G_loss)
        D_loss_metric(D_loss)
        D_real_fake_loss_metric(D_real_fake_loss)
        D_reconstruction_loss_metric(D_reconstruction_loss)

        if step % 500 == 0:
            print(f"Step {step} - "
                  f"G loss {G_loss_metric.result():.4f}, "
                  f"D loss {D_loss_metric.result():.4f}, "
                  f"D realfake loss {D_real_fake_loss_metric.result():.4f}, "
                  f"D recon loss {D_reconstruction_loss_metric.result()}")

    tf.summary.scalar("G_loss/G_loss", G_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_loss", D_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_real_fake_loss", D_real_fake_loss_metric.result(), epoch)
    tf.summary.scalar("D_loss/D_reconstruction_loss", D_reconstruction_loss_metric.result(), epoch)

    print(f"Epoch {epoch} - "
          f"G loss {G_loss_metric.result():.4f}, "
          f"D loss {D_loss_metric.result():.4f}, "
          f"D realfake loss {D_real_fake_loss_metric.result():.4f}, "
          f"D recon loss {D_reconstruction_loss_metric.result()}")

    G_loss_metric.reset_states()
    D_loss_metric.reset_states()

    G.save_weights("./checkpoints/G_checkpoint.h5")
    D.save_weights("./checkpoints/D_checkpoint.h5")
    sle_gan.generate_and_save_images(G, epoch, test_input_for_generation, "logs")
