backpropIterations=(20 40 60 80 100)
rhcIterations=(1000 2000 3000 4000 5000)
saIterations=(500 1000 1500 2000 2500 3000)
gaIterations=(10 20 30 40 50)

count=${#backpropIterations[@]}
i=0

while [ $i -lt $count ]; do
  java -cp ABAGAIL.jar opt.test.PokerTest false \
    ${backpropIterations[i]} \
    ${rhcIterations[i]} \
    ${saIterations[i]} \
    ${gaIterations[i]}
  i=$((i+1))
done
