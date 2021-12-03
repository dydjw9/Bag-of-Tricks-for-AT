python train_cifar.py --model ResNet18 --attack pgd \
                      --lr-schedule piecewise --norm l_inf --epsilon 8 \
                      --epochs 110 --attack-iters 10 --pgd-alpha 2 \
                      --fname auto \
		      --optimizer 'momentum' \
		      --weight_decay 5e-4
                      --batch-size 128 \
		      --BNeval \
