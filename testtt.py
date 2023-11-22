from scipy.stats import t

# degrees of freedom
df = 196

# t-statistic
t_stat = 2.13

# p-value for a two-tailed test
p_value = 2 * (1 - t.cdf(abs(t_stat), df))

print("P-value:", p_value)