count=3

backpropIterations=(100 200 300)
rhcIterations=(1000 2000 3000)
saIterations=(1000 2000 3000)
gaIterations=(10 20 30)

i=0

while [ $i -lt $count ]; do
  java -cp ABAGAIL.jar opt.test.PokerTest false \
    ${backpropIterations[i]} \
    ${rhcIterations[i]} \
    ${saIterations[i]} \
    ${gaIterations[i]}
  i=$((i+1))
done
