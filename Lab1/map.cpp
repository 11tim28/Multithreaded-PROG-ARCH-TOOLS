#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <pthread.h>
#include <mutex>
#include <string>
#include <algorithm>
#include <cctype>

using namespace std;

// Thread arguments
struct ThreadData{
    vector<string>* words;
    size_t start;
    size_t end;
    unordered_map<string, int>* localMap;
};

string normalize(const string& w) {
    string result;
    for (char c : w) {
        if (isalnum((unsigned char)c)) {
            result += tolower((unsigned char)c);
        }
    }
    return result;
}

// Thread Function
void* countWords(void* arg){
    ThreadData* data = (ThreadData*)arg;
    for (size_t i = data->start; i < data->end; i++) {
        string word = normalize((*data->words)[i]);
        if (!word.empty()) {
            ++(*data->localMap)[word];
        }
    }
    return nullptr;
}

void display(vector<string> words){
    for(auto& w : words){
        cout << w << " ";
    }
    cout << endl;
}



int main(int argc, char* argv[]){
    if(argc < 3){
        cerr << "Usage: " << argv[0] << " <filename> <numThreads>\n";
	return 1;
    }
    string filename = argv[1];
    int numThreads = stoi(argv[2]);

    ifstream inFile(filename);
    if(!inFile){
        cerr << "Error opening file: " << filename << endl;
	return 1;
    }
    stringstream buffer;
    buffer << inFile.rdbuf();
    string fileContent = buffer.str();

    vector<string> words;
    string current;
    for(char c : fileContent){
        if(isspace((unsigned char)c) || ispunct((unsigned char)c)){
            if(!current.empty()){
                words.push_back(current);
                current.clear();
            }
        }
        else current += c;
    }
    if(!current.empty()) words.push_back(current);

    size_t length = words.size();
    size_t chunkSize = (length + numThreads - 1) / numThreads;

    vector<unordered_map<string, int>> localMaps(numThreads);
    vector<ThreadData> threadArgs(numThreads);
    vector<pthread_t> threads(numThreads);

    // Launch threads
    for(int i = 0; i < numThreads; i++){
        size_t start = i * chunkSize;
        size_t end = min(start + chunkSize, length);
        threadArgs[i] = {&words, start, end, &localMaps[i]};
        pthread_create(&threads[i], nullptr, countWords, &threadArgs[i]);
    }
    for(int i = 0; i < numThreads; i++){
    	pthread_join(threads[i], nullptr);
    }
    unordered_map<string, int> globalMap;
    for(auto& local : localMaps){
    	for(auto& kv : local){
	    globalMap[kv.first] += kv.second;
	}
    }
    
    cout << "Word counts:\n";
    int printed = 0;
    for(auto& kv : globalMap){
        cout << kv.first << " : " << kv.second << "\n";
	if(++printed > 20) break;
    }
    return 0;

}
