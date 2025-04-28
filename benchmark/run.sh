# time python run_benchmark.py --model=gpt > gptop2.txt
# time python run_benchmark.py --model=moe > moeop2.txt


# for i in $(seq 1 10);
# do
#   time python3 run_benchmark.py --model=gpt --error-lim=5 > run_ops/gpt_no_history_5_$i.txt
# done


# for i in $(seq 1 10);
# do
#   time python3 run_benchmark.py --model=gpt --error-lim=3 > run_ops/gpt_no_history_3_$i.txt
# done


# for i in $(seq 1 10);
# do
#   time python3 run_benchmark.py --model=gpt --error-lim=1 > run_ops/gpt_no_history_1_$i.txt
# done





# for i in $(seq 1 10);
# do
#   time python3 run_benchmark.py --model=moe --error-lim=5 > run_ops/moe_no_history_5_$i.txt
# done


# for i in $(seq 1 10);
# do
#   time python3 run_benchmark.py --model=moe --error-lim=3 > run_ops/moe_no_history_3_$i.txt
# done


# for i in $(seq 1 10);
# do
#   time python3 run_benchmark.py --model=moe --error-lim=1 > run_ops/moe_no_history_1_$i.txt
# done

lst="10";
for i in $lst;
do
  time python3 run_benchmark.py --model=gpt --error-lim=5 > run_ops/gpt_no_history_5_$i.txt
done


lst="9";
for i in $lst;
do
  time python3 run_benchmark.py --model=gpt --error-lim=3 > run_ops/gpt_no_history_3_$i.txt
done

lst="9";
for i in $lst;
do
  time python3 run_benchmark.py --model=gpt --error-lim=1 > run_ops/gpt_no_history_1_$i.txt
done
