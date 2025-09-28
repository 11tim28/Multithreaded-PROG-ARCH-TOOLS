#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

using namespace std;

#define QUEUE_SIZE 5
#define NUM_ITEMS 10

struct BoundedQueue{
    int data[QUEUE_SIZE];
    int head = 0, tail = 0;
    pthread_mutex_t mutex;
    sem_t empty_slots;
    sem_t occupied_slots;
    
    BoundedQueue(){
        pthread_mutex_init(&mutex, nullptr);
        sem_init(&empty_slots, 0, QUEUE_SIZE);
        sem_init(&occupied_slots, 0, 0);
    }
    
    ~BoundedQueue(){
        pthread_mutex_destroy(&mutex);
        sem_destroy(&empty_slots);
        sem_destroy(&occupied_slots);
    }
    
    void push(int value){
        sem_wait(&empty_slots);
        
        pthread_mutex_lock(&mutex);
        data[tail] = value;
        tail = (tail + 1) % QUEUE_SIZE;
        pthread_mutex_unlock(&mutex);
        
        sem_post(&occupied_slots);
    }
    
    int pop(){
        sem_wait(&occupied_slots);
        
        pthread_mutex_lock(&mutex);
        int value = data[head];
        head = (head + 1) % QUEUE_SIZE;
        pthread_mutex_unlock(&mutex);
        
        sem_post(&empty_slots);
        return value;
    }
};

// Stages
BoundedQueue queue12; // Decoder -> Filter
BoundedQueue queue23; // Filter -> Compress

void* decode(void* arg){
    for(int i = 0; i < NUM_ITEMS; i++){
        cout << "Decode: " << i << endl;
        queue12.push(i);
        sleep(1);
    }
    return nullptr;
}

void* filter(void* arg){
    for(int i = 0; i < NUM_ITEMS; i++){
        int val = queue12.pop();
        if(val == -1){
            queue23.push(-1);
            break;
        }
        if(val % 2 != 0){
            cout << "Filter: Dropped " << val << endl;
            continue;
        }
        cout << "Filter: Passed " << val << endl;
        queue23.push(val);
    }
    return nullptr;
}

void* compress(void* arg){
    for(int i = 0; i < NUM_ITEMS; i++){
        int val = queue23.pop();
        int result = val * val;
        cout << "Compress: " << val << " -> " << result << endl;
        //usleep(200); 
    }
    return nullptr;
}


int main(){
    pthread_t tDecode, tFilter, tCompress;
    
    pthread_create(&tDecode, nullptr, decode, nullptr);
    pthread_create(&tFilter, nullptr, filter, nullptr);
    pthread_create(&tCompress, nullptr, compress, nullptr);
    
    pthread_join(tDecode, nullptr);
    pthread_join(tFilter, nullptr);
    pthread_join(tCompress, nullptr);
    
    return 0;
}




