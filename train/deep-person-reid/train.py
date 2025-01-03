import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", help="Train or test")
    parser.add_argument("--weight", default="", help="Path to checkpoint")
    parser.add_argument("--datadir", required=True, help="Path to reid dataset")
    parser.add_argument("--source", default="cuhk03", help="Source dataset")
    parser.add_argument("--target", default="cuhk03", help="Target dataset")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=128, help="Image width")
    parser.add_argument(
        "--batch-size-train", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--batch-size-test", type=int, default=100, help="Test batch size"
    )
    parser.add_argument("--model", default="resnet50", help="Model architecture")
    parser.add_argument("--loss", default="triplet", help="Loss function")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--stepsize", type=int, default=50, help="Scheduler step size")
    parser.add_argument("--max-epoch", type=int, default=60, help="Maximum epochs")
    parser.add_argument(
        "--eval-freq", type=int, default=10, help="Evaluation frequency"
    )
    parser.add_argument("--print-freq", type=int, default=10, help="Print frequency")
    return parser.parse_args()


def main():
    args = parse_args()

    print(args)

    import torchreid  # noqa: E402

    datamanager = torchreid.data.ImageDataManager(
        root=args.datadir,
        sources=args.source,
        targets=args.target,
        height=args.height,
        width=args.width,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        transforms=["random_flip"],
    )

    model = torchreid.models.build_model(
        name=args.model,
        num_classes=datamanager.num_train_pids,
        loss=args.loss,
        pretrained=True,
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=args.lr)

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="single_step", stepsize=args.stepsize
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True
    )

    if args.mode == "train":
        engine.run(
            save_dir=f"log/{args.model}",
            max_epoch=args.max_epoch,
            eval_freq=args.eval_freq,
            print_freq=args.print_freq,
        )
    else:
        # Load weights from checkpoint
        if args.weight:
            torchreid.utils.load_pretrained_weights(model, args.weight)

        engine.run(
            save_dir=f"log/{args.model}",
            max_epoch=args.max_epoch,
            eval_freq=args.eval_freq,
            print_freq=args.print_freq,
            test_only=True,
        )


if __name__ == "__main__":
    main()
