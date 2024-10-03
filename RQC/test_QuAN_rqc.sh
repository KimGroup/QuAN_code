#############################################################################
# bash test_QuAN_rqc.sh cm1 nr nc 0 google: for Fig.3(f)
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

nsymodel=0 # data type that model was trained on, 0 or google
nsydata=$2 # testing data type, 0 or google
d2model=20
batchsize=20
additional=$3

setsize=10000
n_mini=5
h_dim=16
n_head=4
channel=16
prefix='~/QuAN/'

for d1model in 4 6 8 10 12 14 16 18 20
do
    for d1data in 4 6 8 10 12 14 16 18 20
    do
        for run in 0 1 2 3 4 5 6 7
        do
                cmd=" python3 ~/QuAN/RQC/test_QuAN_rqc.py \
                -set $setsize -n_mini $n_mini -d1 $d1data -d2 20 -nr $nr -nc $nc -nsy $nsydata \
                -modelnum $model_name -hdim $h_dim -nhead $n_head -ch $channel -batchsize $batchsize \
                -saveprefix '~/QuAN/Figure/Data_out/g2_saved_models_rqc' -prefix $prefix \
                -wandb_name '${model_name}_A2_p${nsymodel:0:1}_${nr}x${nc}_${d1model}_vs${d2model}_run_${run}' \
                -prev '~/QuAN/Figure/Data_out/g2_saved_models_rqc/${model_name}_A2_p${nsymodel:0:1}_${nr}x${nc}_${d1model}_vs${d2model}_run_${run}/model_va_${nr}x${nc}_F_p${nsymodel}_${d1model}vs${d2model}-${model_name}_set${setsize}_h16nh4_ch16ker2st1_#miniset${n_mini}.pth' \
                $additional "
                echo ""
                echo "d1model=${d1model}, d1data=${d1data}, run=${run} on va."
                echo $cmd
                eval $cmd

                cmd=" python3 ~/QuAN/RQC/test_QuAN_rqc.py \
                -set $setsize -n_mini $n_mini -d1 $d1data -d2 20 -nr $nr -nc $nc -nsy $nsydata \
                -modelnum $model_name -hdim $h_dim -nhead $n_head -ch $channel -batchsize $batchsize \
                -saveprefix '~/QuAN/Figure/Data_out/g2_saved_models_rqc' -prefix $prefix \
                -wandb_name '${model_name}_A2_p${nsymodel:0:1}_${nr}x${nc}_${d1model}_vs${d2model}_run_${run}' \
                -prev '~/QuAN/Figure/Data_out/g2_saved_models_rqc/${model_name}_A2_p${nsymodel:0:1}_${nr}x${nc}_${d1model}_vs${d2model}_run_${run}/model_ta_${nr}x${nc}_F_p${nsymodel}_${d1model}vs${d2model}-${model_name}_set${setsize}_h16nh4_ch16ker2st1_#miniset${n_mini}.pth' \
                $additional "
                echo ""
                echo "d1model=${d1model}, d1data=${d1data}, run=${run} on ta."
                echo $cmd
                eval $cmd
        done
    done
done
