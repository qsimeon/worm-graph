m, single_worm_dataset in dataset.items():
    #     model, log = optimize_model(
    #         dataset=single_worm_dataset["calcium_data"],
    #         model=model,
    #         mask=single_worm_dataset["named_neurons_mask"],
    #         optimizer=optimizer,
    #         start_epoch=epoch,
    #         num_epochs=config.train.epochs,
    #         seq_len=config.train.seq_len,
    #         dataset_size=config.train.dataset_size,
    #     )
    #     logs[worm] = log
    #     epoch = log["epoc