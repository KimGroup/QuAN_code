#############################################################################
# training with simulated data
#############################################################################
model_name='cm1'

nt=$1

if [ $nt -eq 20 ]; then
nr=4
nc=5
fi
if [ $nt -eq 25 ]; then
nr=5
nc=5
fi
if [ $nt -eq 30 ]; then
nr=5
nc=6
fi
if [ $nt -eq 36 ]; then
nr=6
nc=6
fi
nsy=0
depth=$2
batchsize=20
lrn=3.5e-5
add=$3

setsize=10000
n_mini=5
h_dim=16
n_head=4
channel=16
prefix='~/QuAN'
epoch=400

for run in 0 1 2 3 4 5 6 7 8
do
    cmd=" python3 ~/QuAN/RQC/train_QuAN_rqc.py \
    -set $setsize -n_mini $n_mini -d1 $depth -d2 20 -nr $nr -nc $nc -nsy $nsy \
    -modelnum $model_name -hdim $h_dim -nhead $n_head -ch $channel \
    -epoch $epoch -batchsize $batchsize -shuffle_epoch 10 -lrn $lrn -lrstepsize 100 \
    -saveprefix '~/QuAN/Figure/Data_out/g2_saved_models_rqc' -prefix $prefix \
    -wandb_name '${model_name}_A2_p0_${nr}x${nc}_${depth}_vs20_run_${run}' \
    $add "
    echo $cmd
    eval $cmd
done
