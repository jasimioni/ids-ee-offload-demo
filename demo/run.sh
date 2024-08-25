demo/demo-local.py --mq-username remote --mq-password remote --mq-hostname 10.236.89.74 \
                   --mq-queue ee-alexnet --trained-network-file trained_models/AlexNetWithExits_calibrated.pth \
                   --network alexnet --dataset dataset/short_sample.csv --normal-exit1-min-certainty 0.913411 --attack-exit1-min-certainty 0.9 \
                   --normal-exit2-min-certainty 0.926571 --attack-exit2-min-certainty 0.9211 $*
