#############################################################################
# training with simulated data
#############################################################################
model_name=$1
nr=$2
nc=$3
depth=$4
runs=$5
batchsize=$6
lrn=$7
additional=$8

setsize=10000
n_mini=5
h_dim=16
n_head=4
channel=16
prefix='~/QuAN'
epoch=400

for run in $runs
do
    cmd=" python3 ~/QuAN/RQC/train_QuAN_rqc.py \
    -set $setsize -n_mini $n_mini -d1 $depth -d2 20 -nr $nr -nc $nc -nsy 0 \
    -modelnum $model_name -hdim $h_dim -nhead $n_head -ch $channel \
    -epoch $epoch -batchsize $batchsize -shuffle_epoch 10 -lrn $lrn -lrstepsize 100 \
    -saveprefix '~/QuAN/Figure/Data_out/g2_saved_models_rqc' -prefix $prefix \
    -wandb_name '${model_name}_A2_p0_${nr}x${nc}_${depth}_vs20_run_${run}' \
    $additional "
    echo $cmd
    eval $cmd
done
