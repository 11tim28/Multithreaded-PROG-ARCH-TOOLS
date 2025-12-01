#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <stdint.h>

using namespace std;

static const int CHROMOSOME_LENGTH = 20;
static const int POP_SIZE = 128;     // GA evaluation per generation
static const int HEAVY_ITERS = 500;

float evaluate_cost_heavy(const vector<uint8_t>& chromosome)
{
    float sum = 0.0f;

    for (int j = 0; j < CHROMOSOME_LENGTH; j++)
    {
        if (chromosome[j] == 0) continue;

        float acc = 0.0f;
        for (int k = 1; k <= HEAVY_ITERS; k++) {
            float a = sin(j * 0.01f * k);
            float b = cos((j + 1) * 0.02f * k);
            float c = log(1.0f + j * k * 0.0001f);
            acc += a * b + c;
        }
        
        // Add pairwise interactions
        if (j > 0 && chromosome[j-1] == 1)
            acc *= -0.5f;     // bit j conflicts with bit j-1
            
        sum += acc;
    }
    


    return sum;
}

int main()
{
    const uint64_t TOTAL = (1ULL << CHROMOSOME_LENGTH);   // 2^20 = 1,048,576

    vector<uint8_t> chrom(CHROMOSOME_LENGTH);

    float best_cost = -1e9;
    uint64_t best_index = 0;

    vector<float> sampled_curve;
    sampled_curve.reserve(TOTAL / POP_SIZE);

    cout << "Brute forcing " << TOTAL << " chromosomes..." << endl;

    uint64_t next_sample = POP_SIZE;  // sample every 100 evaluations

    for (uint64_t mask = 0; mask < TOTAL; mask++)
    {
        // convert mask ¡÷ chromosome bits
        for (int i = 0; i < CHROMOSOME_LENGTH; i++)
            chrom[i] = (mask >> i) & 1;

        float c = evaluate_cost_heavy(chrom);

        // track global best over all 1,048,576
        if (c > best_cost) {
            best_cost = c;
            best_index = mask;
        }

        // record 1 point every 100 evaluations
        if ((mask + 1) == next_sample) {
            sampled_curve.push_back(best_cost);
            next_sample += POP_SIZE;
        }

        // progress indicator
        if (mask % 100000 == 0)
            cout << "Progress: " << mask << "/" << TOTAL << "\r" << flush;
    }

    cout << "\nDone!" << endl;
    cout << "GLOBAL optimum index: " << best_index << endl;
    cout << "GLOBAL optimum cost:  " << best_cost << endl;

    // ---------------------------------------------
    // Save sampled curve
    // ---------------------------------------------
    ofstream csv("bruteforce_curve.csv");
    csv << "Generation,BestCost\n";

    for (size_t g = 0; g < sampled_curve.size(); g++) {
        csv << g << "," << sampled_curve[g] << "\n";
    }

    csv.close();
    cout << "Saved brute-force curve: bruteforce_curve.csv" << endl;

    return 0;
}
