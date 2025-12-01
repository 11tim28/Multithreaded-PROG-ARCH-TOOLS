set datafile separator ","
set xlabel "Generation"
set ylabel "Best Cost"
set terminal pngcairo size 1200,800 enhanced font 'Arial,14'

###########################################################
# Sorted results plot
###########################################################
set title "GA Convergence Comparison (Sorted)"
set output "ga_convergence_sorted.png"

plot \
    "ga_convergence_cpu_single.csv" using 1:2 with lines title "CPU Single (Sorted)", \
    "ga_convergence_cpu_multi.csv" using 1:2 with lines title "CPU Multi (Sorted)", \
    "ga_convergence_gpu_single.csv" using 1:2 with lines title "GPU Single (Sorted)", \
    "ga_convergence_gpu_multi.csv" using 1:2 with lines title "GPU Multi (Sorted)"

###########################################################
# Random results plot
###########################################################
set title "GA Convergence Comparison (Random)"
set output "ga_convergence_random.png"

plot \
    "ga_convergence_cpu_single.csv" using 1:3 with lines title "CPU Single (Random)", \
    "ga_convergence_cpu_multi.csv" using 1:3 with lines title "CPU Multi (Random)", \
    "ga_convergence_gpu_single.csv" using 1:3 with lines title "GPU Single (Random)", \
    "ga_convergence_gpu_multi.csv" using 1:3 with lines title "GPU Multi (Random)"

###########################################################
# Brute Force results plot
###########################################################
set title "Brute Force Result"
set output "bruteforce_curve.png"

plot \
    "bruteforce_curve.csv" using 1:2 with lines title ""