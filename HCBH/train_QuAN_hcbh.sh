h_dim=16
n_head=4
d1=0
d2=4
model_num=$1
ch=8
add=$2
if [ $model_num == 'c21' ]; then
ch=7
fi

for run in 0 1 2 3 4 5 6 7 8 9
do
	for setsize in 256 128 64 32 16 8 4 2 1
	do
		batchsize=80000/setsize
		cmd="python3 ~/QuAN/HCBH/train_QuAN_hcbh.py \
			-set $setsize -d1 $d1 -d2 $d2 -nr 4 -nc 4 \
			-epoch 500 -batchsize $batchsize -shuffle_epoch 10 \
			-modelnum $model_num -hdim $h_dim -nhead $n_head \
			-dim_outputs 1 -p_outputs 1 -opsc 'yes' \
			-ch $ch -ker 2 -n8 \
			-wandb_name 'boson_${model_num}_${d1}vs${d2}_run_${run}' \
			-saveprefix ~/QuAN/Figure/Data_out/g2_saved_models_hcbh_new/ $add "
		echo $cmd
		eval $cmd
	done
done		
