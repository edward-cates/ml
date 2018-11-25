min_alpha=0.65
max_alpha=0.9
d_alpha=0.05

min_discount=0.6
max_discount=1.0
d_discount=0.1

alpha=$min_alpha
while (( $(awk 'BEGIN {print ("'$alpha'" <= "'$max_alpha'")}') )); do
  discount=$min_discount
  while (( $(awk 'BEGIN {print ("'$discount'" <= "'$max_discount'")}') )); do
    python frozen_lake/q_learning.py $alpha $discount 2>/dev/null
    discount=$(awk "BEGIN {print $discount+$d_discount; exit}")
  done
  alpha=$(awk "BEGIN {print $alpha+$d_alpha; exit}")
done
