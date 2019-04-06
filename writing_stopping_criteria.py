def trainIters(encoder, decoder, n_iters, batch_size=1, print_every=1000, save_every=1000, plot_every=100,
               learning_rate=0.0001):
    start = time.time()
    plot_losses = []
    val_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [sent_pairs[i] for i in range(n_iters)]
    training_pairs = [random.sample(sent_pairs, batch_size) for i in range(n_iters)]

    # training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    patience = 10  # mod Pier

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        # print("################################")
        # print(training_pair)
        input_tensor = training_pair[0][0]
        target_tensor = training_pair[0][1]
        # print("printing tensors for training...")
        # print(input_tensor)
        # print(target_tensor)

        loss = get_train_loss(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                              criterion)
        print_loss_total += loss
        plot_loss_total += loss

        stopping_delta = 0.01  # if improvement is not more than this amount after n tries, exit the loop
        prev_val_loss = 999

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Training loss: %s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                        iter, iter / n_iters * 100, print_loss_avg))

            total_val_loss = 0
            total_val_pairs = len(val_sent_tensor_pairs)

            for itr in range(0, len(val_sent_tensor_pairs)):
                val_input_tensor = val_sent_tensor_pairs[itr][0]
                val_target_tensor = val_sent_tensor_pairs[itr][1]
                # print("Validation record: {0}".format(itr))
                # print(val_sent_pairs[itr])
                val_loss = get_validation_loss(val_input_tensor, val_target_tensor, encoder, decoder, criterion)
                total_val_loss += val_loss

            avg_val_loss = total_val_loss / total_val_pairs
            val_losses.append(avg_val_loss)
            print('Validation loss: %s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                          iter, iter / n_iters * 100, avg_val_loss))

            # mod P_ier
            if abs(avg_val_loss - prev_val_loss) > stopping_delta:
                print(f"No improvement in validation loss, losing patience, saving model : {patience}")
                encoder_save_path = '%s/%s-%d.pth' % (SAVE_PATH, 'encoder', iter)
                print('save encoder weights to ', encoder_save_path)
                torch.save(encoder.state_dict(), encoder_save_path)
                decoder_save_path = '%s/%s-%d.pth' % (SAVE_PATH, 'decoder', iter)
                print('save decoder weights to ', decoder_save_path)
                torch.save(decoder.state_dict(), decoder_save_path)

                patience -= 1

            if patience == 0:  # break out of training
                break

            prev_val_loss = avg_val_loss
            # end mod Pier

            print("##########################################################")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # # save trained encoder and decoder
        # if iter % save_every == 0:
        #     encoder_save_path = '%s/%s-%d.pth' % (SAVE_PATH, 'encoder', iter)
        #     print('save encoder weights to ', encoder_save_path)
        #     torch.save(encoder.state_dict(), encoder_save_path)
        #     decoder_save_path = '%s/%s-%d.pth' % (SAVE_PATH, 'decoder', iter)
        #     print('save decoder weights to ', decoder_save_path)
        #     torch.save(decoder.state_dict(), decoder_save_path)

    showPlot(plot_losses, 'train_plot.png')
    showPlot(val_losses, 'validation_plot.png')

    return plot_losses, val_losses