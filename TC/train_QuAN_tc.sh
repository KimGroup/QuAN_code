h_dim=16
n_head=4

model_num=$1
nr=6
nc=6
p2=0.300

for run in 0 1 2 3 4 5 6 7 8 9
do
    for setsize in 64 32 16 8 4 2 1
    do
    let batchsize=32868/$setsize
    epoch=200
    cmd="python3 ~/QuAN/TC/train_QuAN_tc.py \
    -set $setsize -nr $nr -nc $nc -p1 0.000 -p2 $p2 \
    -epoch $epoch -batchsize $batchsize -shuffle_epoch 10 \
    -modelnum $model_num -hdim $h_dim -nhead $n_head \
    -ch 1 -ker 1 -dim_outputs 1 -p_outputs 1 \
    -wandb_name 'TC_${model_num}_0.000vs${p2}_5_run_${run}' \
    -saveprefix ~/QuAN/Figure/Data_out/g2_saved_models_tc_new/ $2 "
    echo $cmd
    eval $cmd
    done
done
