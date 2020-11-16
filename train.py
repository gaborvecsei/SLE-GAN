import tensorflow as tf

import sle_gan

# For debugging:
# tf.config.experimental_run_functions_eagerly(True)

BATCH_SIZE = 2
EPOCHS = 100

dataset = sle_gan.create_dataset(batch_size=BATCH_SIZE, folder="./dataset")

G = sle_gan.Generator()
sample_G_output = G.initialize()
print(f"[G] output shape: {sample_G_output.shape}")

D = sle_gan.Discriminator()
sample_D_output = D.initialize()
print(f"[D] real_fake output shape: {sample_D_output[0].shape}")
print(f"[D] image output shape{sample_D_output[1].shape}")

G_optimizer = tf.optimizers.Adam(learning_rate=1e-3)
D_optimizer = tf.optimizers.Adam(learning_rate=1e-3)

tb_file_writer = tf.summary.create_file_writer("./logs")
tb_file_writer.set_as_default()

G_loss_metric = tf.keras.metrics.Mean()
D_loss_metric = tf.keras.metrics.Mean()

for epoch in range(EPOCHS):
    for step, image_batch in enumerate(dataset):
        G_loss, D_loss = sle_gan.train_step(G=G,
                                            D=D,
                                            G_optimizer=G_optimizer,
                                            D_optimizer=D_optimizer,
                                            images=image_batch)

        G_loss_metric(G_loss)
        tf.summary.scalar("loss/G_loss", G_loss_metric.result(), epoch)

        D_loss_metric(D_loss)
        tf.summary.scalar("loss/D_loss", G_loss_metric.result(), epoch)

    print(f"Epoch {epoch} - G loss {G_loss_metric.result():.4f}, D loss {D_loss_metric.result():.4f}")

    G_loss_metric.reset_states()
    D_loss_metric.reset_states()
