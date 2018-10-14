package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        // RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        // FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        // fit.train();
        // System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        System.out.println("Randomized Hill Climbing");
        for (int i = 0; i < 10; ++i) {
            int iterations = 25000 * (i + 1);

            long starttime = System.currentTimeMillis();

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations);
            fit.train();
            double value = ef.value(rhc.getOptimal());

            System.out.printf("%d iterations: %s (%dms)%n", iterations, value, (System.currentTimeMillis() - starttime));
        }

        // SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        // fit = new FixedIterationTrainer(sa, 200000);
        // fit.train();
        // System.out.println("SA: " + ef.value(sa.getOptimal()));
        System.out.println("Simulated Annealing");
        // initial temp, cooling
        double[] temps = {1E8, 1E10, 1E12, 1E14, 1E16};
        double[] coolingRates = {0.9, 0.95, 0.99, 0.999};
        double[] maxCombo = {0, 0, 0};
        for (int i = 0; i < temps.length; ++i) {
            for (int j = 0; j < coolingRates.length; ++j) {
                double temp = temps[i], cooling = coolingRates[j];

                SimulatedAnnealing sa = new SimulatedAnnealing(temp, cooling, hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
                fit.train();
                double value = ef.value(sa.getOptimal());

                System.out.printf("initial temp: %f, cooling: %f, value: %s%n", temp, cooling, value);

                if (value > maxCombo[2]) {
                    maxCombo = new double[] {temp, cooling, value};
                }
            }
        }
        double temp = maxCombo[0];
        double cooling = maxCombo[1];
        System.out.printf("best temp: %f, cooling: %f%n", temp, cooling);
        for (int i = 0; i < 10; ++i) {
            int iterations = 25000 * (i + 1);

            long starttime = System.currentTimeMillis();

            SimulatedAnnealing sa = new SimulatedAnnealing(temp, cooling, hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations);
            fit.train();
            double value = ef.value(sa.getOptimal());

            System.out.printf("%d iterations: %s (%dms)%n", iterations, value, (System.currentTimeMillis() - starttime));
        }
        
        // StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        // fit = new FixedIterationTrainer(ga, 1000);
        // fit.train();
        // System.out.println("GA: " + ef.value(ga.getOptimal()));
        System.out.println("Genetic Algorithm");
        // population, toMate, toMutate
        int[] populations = {150, 200, 250};
        int[] mateValues = {50, 100, 150};
        int[] mutateValues = {10, 20, 30};
        maxCombo = new double[] {0, 0, 0, 0};
        for (int i = 0; i < populations.length; ++i) {
            for (int j = 0; j < mateValues.length; ++j) {
                for (int k = 0; k < mutateValues.length; ++k) {
                    int population = populations[i], mates = mateValues[j], mutations = mutateValues[k];

                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(population, mates, mutations, gap);
                    FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
                    fit.train();
                    double value = ef.value(ga.getOptimal());

                    System.out.printf("population: %d, mates: %d, mutations: %d, value: %s%n", population, mates, mutations, value);

                    if (value > maxCombo[3]) {
                        maxCombo = new double[] {population, mates, mutations, value};
                    }
                }
            }
        }
        int population = (int) maxCombo[0];
        int mates = (int) maxCombo[1];
        int mutations = (int) maxCombo[2];
        System.out.printf("best population: %d, mates: %d, mutations: %d%n", population, mates, mutations);
        for (int i = 0; i < 10; ++i) {
            int iterations = 500 * (i + 1);

            long starttime = System.currentTimeMillis();

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(population, mates, mutations, gap);
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, iterations);
            fit.train();
            double value = ef.value(ga.getOptimal());

            System.out.printf("%d iterations: %s (%dms)%n", iterations, value, (System.currentTimeMillis() - starttime));
        }
        
        // MIMIC mimic = new MIMIC(200, 20, pop);
        // fit = new FixedIterationTrainer(mimic, 1000);
        // fit.train();
        // System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
        System.out.println("MIMIC");
        // samples, toKeep
        int[] sampleValues = {100, 150, 200};
        int[] toKeepValues = {2, 4, 6, 8, 10};
        maxCombo = new double[] {0, 0, 0};
        for (int i = 0; i < sampleValues.length; ++i) {
            for (int j = 0; j < toKeepValues.length; ++j) {
                int samples = sampleValues[i], toKeep = toKeepValues[j];

                MIMIC mimic = new MIMIC(samples, toKeep, pop);
                FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 200);
                fit.train();
                double value = ef.value(mimic.getOptimal());

                System.out.printf("samples: %d, toKeep: %d, value: %s%n", samples, toKeep, value);

                if (value > maxCombo[2]) {
                    maxCombo = new double[] {samples, toKeep, value};
                }
            }
        }
        int samples = (int) maxCombo[0];
        int toKeep = (int) maxCombo[1];
        System.out.printf("best samples: %d, toKeep: %d%n", samples, toKeep);
        for (int i = 0; i < 10; ++i) {
            int iterations = 250 * (i + 1);

            long starttime = System.currentTimeMillis();

            MIMIC mimic = new MIMIC(samples, toKeep, pop);
            FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iterations);
            fit.train();
            double value = ef.value(mimic.getOptimal());

            System.out.printf("%d iterations: %s (%dms)%n", iterations, value, (System.currentTimeMillis() - starttime));
        }
    }
}
