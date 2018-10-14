Overview

  The base folder consists of 2 subfolders:
    1. ABAGAIL - all the learning and optimization code provided by the course instructors.
    2. project2 - all the console output logs, graphs, and python scripts used to organize the outputs.

  This documentation assumes familiarity with ABAGAIL since it was provided by the instructors. For ABAGAIL instructions, see https://github.com/pushkar/ABAGAIL


Neural Network

  Poker Test was found at https://gist.github.com/mosdragon/53edf8e69fde531db69e

  And trains a neural network on the given data using backpropagation, randomized hill climbing, simmulated annealing, and a genetic algorithm. 

  The data can be found in project2/data/ (*-dataset.csv)

  After changing to the ABAGAIL directory and compiling using `ant`:

  To perform hyperparameter search for each algorithm:
    java -cp ABAGAIL.jar opt.test.PokerTest true

  To try an increasing number of iterations for each algorithm:
    ./iterations.sh

  In general, the arguments for PokerTest are as follows:
    PokerTest <shouldFindParams: boolean> <backpropIterations: int> <rhcIterations: int> <saIterations: int> <gaIterations: int>


Optimization Problems

  After changing to the ABAGAIL directory and compiling using `ant`:

  Flip flop

    To run and save output:
      java -cp ABAGAIL.jar opt.test.FlipFlopTest > ../project2/output/part2/flipflop.log

    To generate graphs:
      From the project2 directory:
        python3 flipflop.py

      Graphs will be in:
        project2/graphs/part2/

  Knapsack

    To run and save output:
      java -cp ABAGAIL.jar opt.test.KnapsackTest > ../project2/output/part2/knapsack.log

    To generate graphs:
      From the project2 directory:
        python3 knapsack.py

      Graphs will be in:
        project2/graphs/part2/

  Cosine of Sine

    To run and save output:
      java -cp ABAGAIL.jar opt.test.CountOnesTest > ../project2/output/part2/sine.log

    To generate graphs:
      From the project2 directory:
        python3 sine.py

      Graphs will be in:
        project2/graphs/part2/


  Many of the other ABAGAIL tests have been modified to perform hyperparameter search and can be run the same way.
