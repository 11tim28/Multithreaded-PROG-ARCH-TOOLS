#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <queue>
#include <string>
#include <vector>
#include <random>
#include <unistd.h>
#include <algorithm>


using namespace std;

const int POP_SIZE = 10;
const int CHROM_LEN = 10;
const int NUM_OFFSPRING = 10;
const int QUEUE_DEPTH = 10;

const int NUM_SELECTION_THREADS = 2;
const int NUM_CROSSOVER_THREADS = 2;
const int NUM_COST_THREADS = 2;

struct Individual{
    string chromosome;
    int cost;
    bool valid;
};

vector<Individual> parents_pool;
vector<Individual> offspring_pool;
pthread_mutex_t parents_pool_mtx;
pthread_mutex_t offspring_pool_mtx;

template <typename T>
struct BoundedQueue{
    vector<T> buffer;
    pthread_mutex_t mtx;
    sem_t empty_slots;
    sem_t full_slots;
    int capacity;
    int front = 0, rear = 0, count = 0;


    BoundedQueue(int cap): capacity(cap), buffer(cap){
        pthread_mutex_init(&mtx, nullptr);
        sem_init(&empty_slots, 0, cap);
        sem_init(&full_slots, 0, 0);
    }

    // ~BoundedQueue() {
    //     pthread_mutex_destroy(&mtx);
    //     sem_destroy(&empty_slots);
    //     sem_destroy(&full_slots);
    // }

    void push(const T& item){
        sem_wait(&empty_slots);
        pthread_mutex_lock(&mtx);
        buffer[rear] = item;
        rear = (rear + 1) % capacity;
        count++;
        pthread_mutex_unlock(&mtx);
        sem_post(&full_slots);
    }

    T pop(){
        sem_wait(&full_slots);
        pthread_mutex_lock(&mtx);
        T item = buffer[front];
        front = (front + 1) % capacity;
        count--;
        pthread_mutex_unlock(&mtx);
        sem_post(&empty_slots);
        return item;
    }
};

// Queues
BoundedQueue<pair<Individual, Individual>> q12(QUEUE_DEPTH);
BoundedQueue<Individual> q23(QUEUE_DEPTH);

// Random generator
random_device rd;
mt19937 gen(rd());

string randomChromosome(){
    uniform_int_distribution<> d(0, 1);
    string s;
    for(int i = 0; i < CHROM_LEN; i++) s.push_back(d(gen) ? '1' : '0');
    return s;
}

int computeCost(const string& s){
    int score = 0;
    for(int i = 0; i < s.size()-1; i++){
        if(s[i] == '1' && s[i+1] == '1') score++;
    }
    return score + count(s.begin(), s.end(), '1');
}

void* selection(void*){
    // for(int i = 0; i < NUM_OFFSPRING / 2; i++){
    while(true){
        pthread_mutex_lock(&parents_pool_mtx);
        if (parents_pool.size() < 2) {
            pthread_mutex_unlock(&parents_pool_mtx);
            break; // not enough individuals to form a pair
        }

        // sort by cost descending
        sort(parents_pool.begin(), parents_pool.end(),
             [](const Individual &a, const Individual &b) {
                 return a.cost > b.cost;
             });

        // pick first two as a pair
        Individual parent1 = parents_pool[0];
        Individual parent2 = parents_pool[1];

        // remove them from pool
        parents_pool.erase(parents_pool.begin());
        parents_pool.erase(parents_pool.begin());

        pthread_mutex_unlock(&parents_pool_mtx);

        // push the pair into crossover queue
        q12.push({parent1, parent2});

        cout << "[Selection] Picked parents: "
             << parent1.chromosome << "(Cost=" << parent1.cost << ") & "
             << parent2.chromosome << "(Cost=" << parent2.cost << ")\n";
    }
    return nullptr;
}

void* crossover(void*){
    for(int i = 0; i < NUM_OFFSPRING / 2; i++){
        auto parents = q12.pop();
        string p1 = parents.first.chromosome;
        string p2 = parents.second.chromosome;

        int point = CHROM_LEN / 2;
        for(int j = point; j < CHROM_LEN; j++){
            swap(p1[j], p2[j]);
        }
        Individual child1{p1, 0}, child2{p2, 0};
        q23.push(child1);
        q23.push(child2);
        cout << "[Crossover] Produced: " << p1 << " , " << p2 << endl;
        // usleep(150000);
    }
    return nullptr;
}

void* cost(void*){
    for(int i = 0; i < NUM_OFFSPRING; i++){
        Individual child = q23.pop();
        child.cost = computeCost(child.chromosome);

        pthread_mutex_lock(&offspring_pool_mtx);
        offspring_pool.push_back(child);
        pthread_mutex_unlock(&offspring_pool_mtx);

        cout << "[Cost] Child: " << child.chromosome
             << " Cost=" << child.cost << endl;
        // usleep(120000);
    }
    return nullptr;
}

int main(){
    // Init parents_pool
    pthread_mutex_init(&parents_pool_mtx, nullptr);
    for(int i = 0; i < POP_SIZE; i++){
        string chr = randomChromosome();
        parents_pool.push_back({chr, computeCost(chr), true});
    }
    float parents_avg = 0;
    float offspring_avg = 0;
    cout << "Initial parents_pool:\n";
    for (auto& ind : parents_pool) {
        cout << ind.chromosome << " (Cost=" << ind.cost << ")\n";
        parents_avg += ind.cost;
    }
    parents_avg /= parents_pool.size();
    // cout << "Parents Average Cost: " << parents_avg << endl;
    cout << "-------------------------\n";

    // Create Threads
    pthread_t selection_threads[NUM_SELECTION_THREADS];
    pthread_t crossover_threads[NUM_CROSSOVER_THREADS];
    pthread_t cost_threads[NUM_COST_THREADS];
    for(int i = 0; i < NUM_SELECTION_THREADS; i++) pthread_create(&selection_threads[i], nullptr, selection, nullptr);
    for(int i = 0; i < NUM_CROSSOVER_THREADS; i++) pthread_create(&crossover_threads[i], nullptr, crossover, nullptr);
    for(int i = 0; i < NUM_COST_THREADS; i++) pthread_create(&cost_threads[i], nullptr, cost, nullptr);


    // Join
    for(int i = 0; i < NUM_SELECTION_THREADS; i++) pthread_join(selection_threads[i], nullptr);
    for(int i = 0; i < NUM_CROSSOVER_THREADS; i++) pthread_join(crossover_threads[i], nullptr);
    for(int i = 0; i < NUM_COST_THREADS; i++) pthread_join(cost_threads[i], nullptr);

    cout << "-------------------------\nFinal offspring_pool Size: " << offspring_pool.size() << endl;
    for (auto& ind : offspring_pool) {
        cout << ind.chromosome << " (Cost=" << ind.cost << ")\n";
        offspring_avg += ind.cost;
    }
    offspring_avg /= offspring_pool.size();
    cout << "Parents Average Cost: " << parents_avg << endl;
    cout << "Offspring Average Cost: " << offspring_avg << endl;

    return 0;
}


