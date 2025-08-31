
# This script performs mixed model analysis using the AI-REML algorithm.
# It handles missing phenotype values and can accommodate multiple random effects.

using DelimitedFiles, Statistics
using LinearAlgebra, Printf
using Missings

## File I/O
# This section handles all data reading and writing operations.
println("Loading data files...")

# Read all data files and store them in variables
y_str = readdlm("C://Users//patwa//Documents//phenotypeNA.txt", ',', header=false, String)
M = readdlm("C://Users//patwa//Documents//QTLMAS2010gen.txt", ',', header=false)
X = readdlm("C://Users//patwa//Documents//fixedcov.txt", ',', header=false)

# Read additional random effects data if the file exists
R_data = nothing
try
    R_data = readdlm("C://Users//patwa//Documents//extrarandeff.txt", ',', header=false, skipstart=0)
    if size(R_data, 2) == 1 && all(iszero, R_data)
        println("Random effects file found, but no effects detected (column of zeros).")
        R_data = nothing
    else
        println("Random effects file found and loaded.")
    end
catch e
    if isa(e, SystemError)
        println("Random effects file not found. Running GBLUP model only.")
    else
        rethrow(e)
    end
end

println("...Done loading data.")

## Data Preprocessing and Model Setup
# This section prepares the data matrices for the AI-REML algorithm.
println("\nSetting up model matrices...")

# Manually convert the string matrix to handle missing values
y = Array{Union{Float64, Missing}}(undef, size(y_str))
for i in eachindex(y_str)
    if uppercase(strip(y_str[i])) == "NA"
        y[i] = missing
    else
        try
            y[i] = parse(Float64, y_str[i])
        catch e
            println("Error parsing value at index $i: $(y_str[i])")
            rethrow(e)
        end
    end
end

n = size(M)[1] # Number of individuals
l = size(M)[2] # Number of markers

# --- FIXED EFFECTS DATA PROCESSING ---
# Check for and add intercept, transform categorical variables.
println("\nProcessing fixed effects data...")

# Check for intercept (first column of all ones)
has_intercept = size(X, 2) > 0 && all(X[:, 1] .== 1.0)
if !has_intercept
    X = hcat(ones(size(X, 1), 1), X)
    println("Intercept column added to the fixed effects matrix.")
end

# Process the remaining columns of X (starting from the second column if an intercept was added)
processed_X = X[:, 1:1] # Start with the intercept column
for col_idx in 2:size(X, 2)
    current_col = X[:, col_idx]
    unique_levels = unique(current_col)
    n_levels = length(unique_levels)

    if n_levels == 2
        # Binary variable: transform to 0/1
        if 0.0 ∉ unique_levels
            transformed_col = current_col .== unique_levels[1]
        else
            transformed_col = current_col .== 1.0
        end
        processed_X = hcat(processed_X, transformed_col)
        println("Column $col_idx (binary) transformed to 0/1.")
    elseif n_levels > 2
        # Factor variable: transform to level indicators (drop first level)
        level_indicators = zeros(size(X, 1), n_levels - 1)
        # Sort unique levels to ensure consistent ordering
        sorted_levels = sort(unique_levels)
        for i in 2:n_levels
            level = sorted_levels[i]
            level_indicators[:, i-1] = current_col .== level
        end
        processed_X = hcat(processed_X, level_indicators)
        println("Column $col_idx (factor with $n_levels levels) transformed to indicator variables.")
    else # Continuous variable (n_levels == 1)
        processed_X = hcat(processed_X, current_col)
        println("Column $col_idx (continuous/single level) left as is.")
    end
end

X = processed_X
println("...Done processing fixed effects.")
# --- END FIXED EFFECTS DATA PROCESSING ---


# Calculate the Genomic Relationship Matrix (GRM) using the VanRaden (2008) method
N = M .- 1
p = sum(M; dims=1) ./ (n * 2)
Z_gen = N .- 2 * (p .- 0.5)
G = (Z_gen * Z_gen') ./ (2 * sum(p .* (1 .- p)))

# Prepare all kinship/relationship matrices for the model
K_matrices_full = [G]
if R_data !== nothing
    for i in 1:size(R_data, 2)
        rand_effect_levels = R_data[:, i]
        unique_levels = unique(rand_effect_levels)
        Z_rand = zeros(size(R_data, 1), length(unique_levels))
        for j in 1:length(unique_levels)
            Z_rand[rand_effect_levels .== unique_levels[j], j] .= 1.0
        end
        K_rand_full = Z_rand * Z_rand'
        push!(K_matrices_full, K_rand_full)
    end
    println("...Done. Found $(length(K_matrices_full)-1) additional random effects.")
end

## AI-REML Algorithm
# This section contains the core mixed model analysis logic.
if any(ismissing, y)
    println("Missing values detected in phenotype data. Adjusting analysis for missing data.")

    # Missing Data Handling
    observed_idx_mat = .!ismissing.(y)
    observed_idx = vec(observed_idx_mat)
    n_obs = sum(observed_idx)
    y_obs = collect(skipmissing(y))
    X_obs = X[observed_idx, :]

    # Subset all random effect matrices based on observed individuals
    K_matrices_obs = []
    for K_mat in K_matrices_full
        K_obs = K_mat[observed_idx, observed_idx]
        push!(K_matrices_obs, K_obs)
    end

    # AI-REML Loop for missing data
    num_vc = length(K_matrices_obs) + 1
    σ_sq_vec = ones(num_vc)
    X_obs_t = X_obs'
    max_iter = 100
    convergence_threshold = 1e-6
    println("Starting AI-REML algorithm...")
    header_string = "Iter |    σ²_G    |"
    for i in 2:length(K_matrices_obs)
        header_string *= " σ²_rand$(i-1) |"
    end
    header_string *= "   σ²_e    | Log-Likelihood | Change"
    println(header_string)
    println("-"^(length(header_string)-4))

    AI = zeros(num_vc, num_vc)
    P_final = zeros(n_obs, n_obs)
    log_likelihood = 0.0

    for iter in 1:max_iter
        old_σ_sq_vec = copy(σ_sq_vec)
        V_obs = zeros(n_obs, n_obs)
        for i in 1:length(K_matrices_obs)
            V_obs += σ_sq_vec[i] * K_matrices_obs[i]
        end
        V_obs += σ_sq_vec[num_vc] * I
        V_obs_inv = inv(V_obs)
        X_t_V_inv_X = X_obs_t * V_obs_inv * X_obs
        P_obs = V_obs_inv - V_obs_inv * X_obs * inv(X_t_V_inv_X) * X_obs_t * V_obs_inv
        if iter == max_iter || all(abs.(old_σ_sq_vec - σ_sq_vec) .< convergence_threshold)
            P_final = P_obs
        end
        log_likelihood = -0.5 * (logdet(V_obs) + logdet(X_t_V_inv_X) + (y_obs' * P_obs * y_obs)[1])
        gradients = zeros(num_vc)
        for i in 1:length(K_matrices_obs)
            gradients[i] = -0.5 * tr(P_obs * K_matrices_obs[i]) + 0.5 * (y_obs' * P_obs * K_matrices_obs[i] * P_obs * y_obs)[1]
        end
        gradients[num_vc] = -0.5 * tr(P_obs) + 0.5 * (y_obs' * P_obs * P_obs * y_obs)[1]
        for i in 1:num_vc
            for j in i:num_vc
                K_i = (i <= length(K_matrices_obs)) ? K_matrices_obs[i] : I
                K_j = (j <= length(K_matrices_obs)) ? K_matrices_obs[j] : I
                AI[i,j] = 0.5 * tr(P_obs * K_i * P_obs * K_j)
                AI[j,i] = AI[i,j]
            end
        end
        delta = AI \ gradients
        σ_sq_vec += delta
        σ_sq_vec = max.(σ_sq_vec, 1e-8)
        change = sum(abs.(delta))
        @printf("%4d | %8.4f |", iter, σ_sq_vec[1])
        for i in 2:num_vc-1
            @printf(" %8.4f |", σ_sq_vec[i])
        end
        @printf(" %8.4f | %14.4f | %e\n", σ_sq_vec[num_vc], log_likelihood, change)
        if change < convergence_threshold
            println("\nAlgorithm converged.")
            break
        end
    end

    V_final_obs = zeros(n_obs, n_obs)
    for i in 1:length(K_matrices_obs)
        V_final_obs += σ_sq_vec[i] * K_matrices_obs[i]
    end
    V_final_obs += σ_sq_vec[num_vc] * I
    V_final_obs_inv = inv(V_final_obs)
    X_t_V_inv_X = X_obs' * V_final_obs_inv * X_obs
    β_hat_final = inv(X_t_V_inv_X) * X_obs' * V_final_obs_inv * y_obs
    β_cov = inv(X_t_V_inv_X)
    β_se = sqrt.(diag(β_cov))

    # Correctly calculate the GEBVs and other random effects
    G_full_obs_cols = G[:, observed_idx]

    # Predict the genomic random effects (GEBVs)
    g_hat_all = σ_sq_vec[1] * G_full_obs_cols * P_final * (y_obs - X_obs * β_hat_final)

    # Predict additional random effects and sum them up with the GEBVs
    if length(K_matrices_full) > 1
        u_hat_parts = []
        push!(u_hat_parts, g_hat_all)
        for i in 2:length(K_matrices_full)
            K_full_obs_cols = K_matrices_full[i][:, observed_idx]
            u_rand_hat = σ_sq_vec[i] * K_full_obs_cols * P_final * (y_obs - X_obs * β_hat_final)
            push!(u_hat_parts, u_rand_hat)
        end
        u_hat_all = sum(u_hat_parts)
    else
        u_hat_all = g_hat_all
    end

    PEV_all = σ_sq_vec[1] * G - σ_sq_vec[1]^2 * G_full_obs_cols * P_final * G_full_obs_cols'
    u_se_all = sqrt.(diag(PEV_all))
    p_fixed = size(X_obs, 2)

else # No missing values in y
    println("No missing values detected. Using the full dataset for analysis.")
    y_obs = y
    n_obs = n
    X_obs = X
    K_matrices_obs = K_matrices_full
    P_final = zeros(n, n)

    # AI-REML Loop for non-missing data
    num_vc = length(K_matrices_obs) + 1
    σ_sq_vec = ones(num_vc)
    X_obs_t = X_obs'
    max_iter = 100
    convergence_threshold = 1e-6
    println("Starting AI-REML algorithm...")
    header_string = "Iter |    σ²_G    |"
    for i in 2:length(K_matrices_obs)
        header_string *= " σ²_rand$(i-1) |"
    end
    header_string *= "   σ²_e    | Log-Likelihood | Change"
    println(header_string)
    println("-"^(length(header_string)-4))

    AI = zeros(num_vc, num_vc)
    log_likelihood = 0.0
    for iter in 1:max_iter
        old_σ_sq_vec = copy(σ_sq_vec)
        V = zeros(n_obs, n_obs)
        for i in 1:length(K_matrices_obs)
            V += σ_sq_vec[i] * K_matrices_obs[i]
        end
        V += σ_sq_vec[num_vc] * I
        V_inv = inv(V)
        X_t_V_inv_X = X_obs' * V_inv * X_obs
        P = V_inv - V_inv * X_obs * inv(X_t_V_inv_X) * X_obs_t * V_inv
        if iter == max_iter || all(abs.(old_σ_sq_vec - σ_sq_vec) .< convergence_threshold)
            P_final = P
        end
        log_likelihood = -0.5 * (logdet(V) + logdet(X_t_V_inv_X) + (y_obs' * P * y_obs)[1])
        gradients = zeros(num_vc)
        for i in 1:length(K_matrices_obs)
            gradients[i] = -0.5 * tr(P * K_matrices_obs[i]) + 0.5 * (y_obs' * P * K_matrices_obs[i] * P * y_obs)[1]
        end
        gradients[num_vc] = -0.5 * tr(P) + 0.5 * (y_obs' * P * P * y_obs)[1]
        for i in 1:num_vc
            for j in i:num_vc
                K_i = (i <= length(K_matrices_obs)) ? K_matrices_obs[i] : I
                K_j = (j <= length(K_matrices_obs)) ? K_matrices_obs[j] : I
                AI[i,j] = 0.5 * tr(P * K_i * P * K_j)
                AI[j,i] = AI[i,j]
            end
        end
        delta = AI \ gradients
        σ_sq_vec += delta
        σ_sq_vec = max.(σ_sq_vec, 1e-8)
        change = sum(abs.(delta))
        @printf("%4d | %8.4f |", iter, σ_sq_vec[1])
        for i in 2:num_vc-1
            @printf(" %8.4f |", σ_sq_vec[i])
        end
        @printf(" %8.4f | %14.4f | %e\n", σ_sq_vec[num_vc], log_likelihood, change)
        if change < convergence_threshold
            println("\nAlgorithm converged.")
            break
        end
    end

    V_final = zeros(n, n)
    for i in 1:length(K_matrices_obs)
        V_final += σ_sq_vec[i] * K_matrices_obs[i]
    end
    V_final += σ_sq_vec[num_vc] * I
    V_final_inv = inv(V_final)
    X_t_V_inv_X = X_obs' * V_final_inv * X_obs
    β_hat_final = inv(X_t_V_inv_X) * X_obs' * V_final_inv * y_obs
    β_cov = inv(X_t_V_inv_X)
    β_se = sqrt.(diag(β_cov))

    # Predict the genomic random effects (GEBVs)
    g_hat_all = σ_sq_vec[1] * K_matrices_obs[1] * P_final * (y_obs - X_obs * β_hat_final)

    # Predict additional random effects and sum them up with the GEBVs
    if length(K_matrices_full) > 1
        u_hat_parts = []
        push!(u_hat_parts, g_hat_all)
        for i in 2:length(K_matrices_full)
            u_rand_hat = σ_sq_vec[i] * K_matrices_full[i] * P_final * (y_obs - X_obs * β_hat_final)
            push!(u_hat_parts, u_rand_hat)
        end
        u_hat_all = sum(u_hat_parts)
    else
        u_hat_all = g_hat_all
    end

    PEV_all = σ_sq_vec[1] * K_matrices_obs[1] - σ_sq_vec[1]^2 * K_matrices_obs[1] * P_final * K_matrices_obs[1]
    u_se_all = sqrt.(diag(PEV_all))
    p_fixed = size(X, 2)
end

## Final Prediction and Output
# This section prints the results to the console and saves them to a file.

# Predict phenotypes for ALL individuals.
y_hat_all = X * β_hat_final + u_hat_all

z_score = 1.96
AI_inv = inv(AI)
std_devs = sqrt.(diag(AI_inv))

println("\nFinal AI-REML Estimates:")
println("-------------------------")
for i in 1:num_vc
    if i == 1
        vc_name = "Genomic (σ²_G)"
    elseif i <= length(K_matrices_full)
        vc_name = "Additional Random Effect $(i-1)"
    else
        vc_name = "Residual (σ²_e)"
    end
    ci = (σ_sq_vec[i] - z_score * std_devs[i], σ_sq_vec[i] + z_score * std_devs[i])
    println("Estimated ", vc_name, ": ", @sprintf("%.4f", σ_sq_vec[i]), " (95%% CI: ", @sprintf("%.4f", ci[1]), " to ", @sprintf("%.4f", ci[2]), ")")
end

println("\nEstimated Fixed Effects:")
println("------------------------")
for i in 1:length(β_hat_final)
    β_ci_i = (β_hat_final[i] - z_score * β_se[i], β_hat_final[i] + z_score * β_se[i])
    @printf("Fixed Effect %d: %8.4f (95%% CI: %8.4f to %8.4f)\n", i, β_hat_final[i], β_ci_i[1], β_ci_i[2])
end

println("\nEstimated Genomic Estimated Breeding Values (GEBVs):")
println("-------------------------------------------------------")
for j in 1:5
    g_ci_i = (g_hat_all[j] - z_score * u_se_all[j], g_hat_all[j] + z_score * u_se_all[j])
    @printf("Individual %d: %8.4f (95%% CI: %8.4f to %8.4f)\n", j, g_hat_all[j], g_ci_i[1], g_ci_i[2])
end
println("")

k = p_fixed + num_vc
aic = -2 * log_likelihood + 2 * k
bic = -2 * log_likelihood + k * log(n_obs)
aicc = aic + (2*k*(k+1))/(n_obs-k-1)

println("\nModel Fit Criteria:")
println("--------------------")
@printf("Log-Likelihood: %14.4f\n", log_likelihood)
@printf("AIC: %14.4f\n", aic)
@printf("BIC: %14.4f\n", bic)
@printf("AICc: %14.4f\n", aicc)

σ_g_sq = σ_sq_vec[1]
σ_p_sq = sum(σ_sq_vec)
h_sq = σ_g_sq / σ_p_sq
d = zeros(num_vc)
d[1] = (σ_p_sq - σ_g_sq) / σ_p_sq^2
d[2:end] .= -σ_g_sq / σ_p_sq^2
var_h_sq = (d' * AI_inv * d)[1]
std_h_sq = sqrt(var_h_sq)
h_sq_ci = (h_sq - z_score * std_h_sq, h_sq + z_score * std_h_sq)

println("\nHeritability Estimates:")
println("-----------------------")
println("Heritability (h²): ", @sprintf("%.4f", h_sq), " (95% CI: ", @sprintf("%.4f", h_sq_ci[1]), " to ", @sprintf("%.4f", h_sq_ci[2]), ")")

# Save all GEBVs to a text file
open("C://Users//patwa//Documents//GEBV_output.txt", "w") do io
    @printf(io, "Individual\tGEBV\tLower_CI\tUpper_CI\n")
    for i in 1:n
        ci_lower = g_hat_all[i] - z_score * u_se_all[i]
        ci_upper = g_hat_all[i] + z_score * u_se_all[i]
        @printf(io, "%d\t%.4f\t%.4f\t%.4f\n", i, g_hat_all[i], ci_lower, ci_upper)
    end
end
println("\nGEBVs and their 95% CIs for all individuals have been saved to C://Users//patwa//Documents//GEBV_output.txt")


## SAVE PREDICTED PHENOTYPES TO A FILE

open("C://Users//patwa//Documents//yhat_output.txt", "w") do io
    @printf(io, "Individual\tPredicted_Phenotype\n")
    for i in 1:n
            @printf(io, "%d\t%.4f\n", i[1], y_hat_all[i])
    end
end
println("\nPredicted phenotype values for all individuals have been saved to C://Users//patwa//Documents//yhat_output.txt")
