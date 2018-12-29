#!/bin/bash
# rm -rf learn_logs learn_model
source activate test36
# python dqn.py  > nohup.out 2>&1 
# python dqn.py --modelname full  > nohup_full.out 2>&1
# python dqn.py --modelname one_y  --learnrate 1e-05 --globalstep 0 > nohup_one_y.out 2>&1 &
# python dqn.py --modelname one_space  --learnrate 1e-05 --globalstep 0 > nohup_one_space.out 2>&1 &
# python dqn.py --modelname one --learnrate 1e-05 --globalstep 0 > nohup_one.out 2>&1 &
python dqn.py --modelname one_attent60 --learnrate 1e-08 --normal 1e-03 --globalstep 0 --dropout 1.0 --sub_fix 1.0 > nohup_one_attent60-normal3.out 2>&1 &
python dqn.py --modelname one_attent60 --learnrate 1e-08 --normal 1e-04 --globalstep 0 --dropout 1.0 --sub_fix 1.0 > nohup_one_attent60-normal4.out 2>&1 &
python dqn.py --modelname one_attent60 --learnrate 1e-08 --normal 1e-04 --globalstep 0 --dropout 0.7 --sub_fix 0.7 > nohup_one_attent60-0.7.out 2>&1 &
python dqn.py --modelname one_attent60 --learnrate 1e-08 --normal 1e-04 --globalstep 0 --dropout 0.5 --sub_fix 0.5 > nohup_one_attent60-0.5.out 2>&1 &
python dqn.py --modelname one_attent --learnrate 1e-08 --normal 1e-03 --globalstep 0 --dropout 1.0 --sub_fix 1.0 > nohup_one_attent-normal3.out 2>&1 &
python dqn.py --modelname one_attent --learnrate 1e-08 --normal 1e-04 --globalstep 0 --dropout 1.0 --sub_fix 1.0 > nohup_one_attent-normal4.out 2>&1 &
python dqn.py --modelname one_attent --learnrate 1e-08 --normal 1e-04 --globalstep 0 --dropout 0.7 --sub_fix 0.7 > nohup_one_attent-0.7.out 2>&1 &
python dqn.py --modelname one_attent --learnrate 1e-08 --normal 1e-04 --globalstep 0 --dropout 0.5 --sub_fix 0.5 > nohup_one_attent-0.5.out 2>&1 &

