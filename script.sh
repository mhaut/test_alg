

for idte in 0 1 2 3 4
do
    for dataset in IP KSC UP
    do
        for tper in 0.01 0.05 0.10 0.15 0.20
        do
            python -u svm.py --dataset $dataset --idtest $idte --tr_percent $tper
            python -u rf.py --dataset $dataset --idtest $idte --tr_percent $tper
            python -u mlr.py --dataset $dataset --idtest $idte --tr_percent $tper
        done
    done
    python -u svm.py --dataset UH --idtest $idte
    python -u rf.py --dataset UH --idtest $idte
    python -u mlr.py --dataset UH --idtest $idte
done




for idte in 0 1 2 3 4
do
    for dataset in IP KSC UP
    do
        for dataset in 0.01 0.05 0.10 0.15 0.20
        do
            python -u mlp.py --dataset $dataset --idtest $idte --tr_percent $tper --use_val
            python -u recurrent.py --dataset $dataset --idtest $idte --tr_percent $tper --type_recurrent RNN --use_val
            python -u recurrent.py --dataset $dataset --idtest $idte --tr_percent $tper --type_recurrent GRU --use_val
            python -u recurrent.py --dataset $dataset --idtest $idte --tr_percent $tper --type_recurrent LSTM --use_val
            python -u cnn1d.py --dataset $dataset --idtest $idte --tr_percent $tper --use_val
            python -u cnn2d.py --dataset $dataset --idtest $idte --tr_percent $tper --components 1 --use_val
            python -u cnn2d.py --dataset $dataset --idtest $idte --tr_percent $tper --components 40 --use_val
            python -u cnn3d.py --dataset $dataset --idtest $idte --tr_percent $tper --components 40 --use_val
        done
    done
    python -u mlp.py --dataset UH --idtest $idte --use_val
    python -u recurrent.py --dataset UH --idtest $idte --type_recurrent RNN --use_val
    python -u recurrent.py --dataset UH --idtest $idte --type_recurrent GRU --use_val
    python -u recurrent.py --dataset UH --idtest $idte --type_recurrent LSTM --use_val
    python -u cnn1d.py --dataset UH --idtest $idte --use_val
    python -u cnn2d.py --dataset UH --idtest $idte --components 1 --use_val
    python -u cnn2d.py --dataset UH --idtest $idte --components 40 --use_val
    python -u cnn3d.py --dataset UH --idtest $idte --components 40 --use_val
done
