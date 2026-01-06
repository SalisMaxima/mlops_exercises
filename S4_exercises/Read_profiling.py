import pstats

p = pstats.Stats("optimized_final.prof")
p.sort_stats("cumulative").print_stats(10)


p = pstats.Stats("optimized.prof")
p.sort_stats("cumulative").print_stats(10)

p = pstats.Stats("profile.prof")
p.sort_stats("cumulative").print_stats(10)
