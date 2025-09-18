###################################################################################
# Read QTLMAS2010 data including genomic markers, breeding values and sex indicator
###################################################################################

using DelimitedFiles, Statistics, Printf
tbvdata = readdlm("C://Users//patwa//Documents//tbv.txt",',')
gender = tbvdata[:,2]
s = gender .==1 #Male/Sire indicator variable
d = gender .==0 #Female/Dam indicator variable
uhat = tbvdata[:,5] #Use simulated true breeding values as BVs (uhat)
X = readdlm("C://Users//patwa//Documents//QTLMAS2010gen.txt",',') #Markers (SNPs in 0,1,2 format)

n = size(X)[1] #Number of individuals

#######################################
#Calculate G-matrix following VanRaden1
#######################################

l = size(X)[2] #Number of markers
N = X .- 1
p = sum(X;dims=1)./(n*2)
Z = N .-  2*(p.-0.5)
G = (Z*Z')./(2*sum(p.*(1 .-p)))

#Correction for inbreeding, non-positive definite and not used here
#d = diag(G)
#Gstd = zeros(n, n)
#for i in 1:n
#    for j in 1:n
#        Gstd[i, j] = G[i,j]./sqrt(G[i,i]*G[j,j])
#    end
#    #Gstd[i,i] = d[i]
#end


#Check if G positive definite, not needed with VanRaden1
#using Bigsimr
#isposdef(G)
#Avoid if G already is pd
#Gpd = cor_nearPD(Gstd)
#isposdef(Gpd)


###############################################
#Perform genomic optimum contribution selection
###############################################

using JuMP, LinearAlgebra, COSMO
mG = JuMP.Model(COSMO.Optimizer);
theta = 0.01 #The coancestry/inbreeding constraint
@variable(mG, c[1:n] >= 0) #The optimum contribution variable with positivity constraint
@objective(mG, Max, dot(uhat,c)); #The objective finction to be maximized
@constraint(mG, sum(d[i]*c[i] for i in 1:n) == 0.5); #The sum constraint on females/dams
@constraint(mG, sum(s[i]*c[i] for i in 1:n) == 0.5); #The sum constraint on males/sires
@constraints(mG, begin
  sum(c[i] * G[i, j] * c[j] for i = 1:n, j = 1:n)/2 <= theta
end) #The quadratic constraint using the genomic relationship matrix

#Run the model
JuMP.optimize!(mG);

#Extract OC values from JuMP
contrG = JuMP.value.(c)

#Set small values of OC to zero and extract selected individuals
rho = 0.0001 #OC selection threshold, data dependent
selcontrind = contrG.>rho
cosel = contrG[selcontrind]
uhatsel = uhat[selcontrind]
indseq = 1:n
selind = indseq[selcontrind]
nsel = size(selind)[1]
damselindG = selind[d[selind].==1] #Selected females/dams
sireselindG = selind[s[selind].==1] #Selected males/sires
nd = size(damselindG)[1] #Number of selected females/dams
ns = size(sireselindG)[1] #Number of selected males/sires

#Plot the OC values against the BVs with separate lables
using Plots
scatter(uhat,contrG,label="No OC")
xlabel!("BV")
ylabel!("OC")
scatter!(uhat[damselindG],contrG[damselindG],label="Dam OC")
scatter!(uhat[sireselindG],contrG[sireselindG],label="Sire OC")
savefig("C://Users//patwa//Documents//QTLMAS2010OCG1.png")


##########################################################################################
#Mating allocations using Mixed Interger Quadratic Programming prop to OCS, binary matings
##########################################################################################

#Extract the H matrix (reduced G matrix) of the selected individuals
H = zeros(nd, ns)
for i in 1:nd
    for j in 1:ns
        H[i, j] = G[damselindG[i],sireselindG[j]]
    end
end

kappad = 3 #Max number of matings per female
kappas = 0 #Min number of matings per male
c_s = contrG[sireselindG] #Extract the OC values for selected sires
c_d = contrG[damselindG] #Extract the OC values for selected dams
using SCIP
mall1 = Model(SCIP.Optimizer)
@variable(mall1, MB[1:nd, 1:ns], Bin); #Variable allowing only binary (0/1) matings
#Constraint max number of matings for the selected females 
@constraint(mall1, fem_mateB[i in 1:nd], sum(MB[i, j] for j in 1:ns) <= kappad);
#Constraint min number of matings for the selected males 
@constraint(mall1, male_mateB[j in 1:ns], sum(MB[i, j] for i in 1:nd) >= kappas);
@objective(mall1, Min, c_d'*(H .* MB)*c_s); #Minimize coancestry of selected matings

#Run the model and check convergence
optimize!(mall1)
assert_is_solved_and_feasible(mall1)
println("Optimal value: ", objective_value(mall1))

#Extract the mating scheme M (here called MB due to the binary matings)
matselB = round.(Int,(value.(MB)))
nmatdamB = round.(Int,(sum.(eachrow(matselB)))) #Number of matings for each female
nmatsireB = round.(Int,(sum.(eachcol(matselB)))) #Number of matings for each male
siremanzeroB = nmatsireB[nmatsireB.==0] #Sires with zero matings
siremaebvzeroB = uhat[sireselindG][nmatsireB.==0] #BVs of sires with zero matings
siremaindB = sireselindG[nmatsireB.>0] #Original indices of sires with matings > 0
siremanB = nmatsireB[nmatsireB.>0] #Number of matings of sires with matings > 0
siremaebvB = uhat[sireselindG][nmatsireB.>0] #BVs of sires with matings > 0
#print([siremaindB,siremanB,sirematbvB])

MAindnozeroB = findall(!iszero, matselB) #Extract non-zero matings

#Make a dictionary of the extracted matings
MAindB = IdDict()
for i in 1:nd
    for j in 1:ns
        MAindB[i,j] = [damselindG[i],sireselindG[j]]
    end
end

#Print out the matings with original id and number of matings[[dam,sire],number of matings]
for i in 1:size(MAindnozeroB)[1]
  println([MAindB[MAindnozeroB[i][1],MAindnozeroB[i][2]],matselB[MAindnozeroB[i][1],MAindnozeroB[i][2]]])
end

#Print out the matings to a text file
open("C://Users//patwa//Documents//QTLMAS2010MA1B.txt", "w") do io
    # Write the column headers
    @printf(io, "%s\t%s\t%s\n", "Dam", "Sire", "Nmat")

    # Loop to write the data
    for i in 1:size(MAindnozeroB)[1]
        matingsB = MAindB[MAindnozeroB[i][1], MAindnozeroB[i][2]]
        nmatselB = matselB[MAindnozeroB[i][1], MAindnozeroB[i][2]]
        
        # Write the data for each row
        @printf(io, "%d\t%d\t%d\n", matingsB[1], matingsB[2], nmatselB)
    end
end

#Plot the number of matings against BVs for the binary setting
scatter(siremaebvzeroB,siremanzeroB,label="Sire zero matings")
xlabel!("BV")
ylabel!("Number of matings")
scatter!(uhat[damselindG],nmatdamB,label="Dam matings")
scatter!(siremaebvB,siremanB,label="Sire matings")
savefig("C://Users//patwa//Documents//QTLMAS2010MA1B.png")

# Calculate the total number of matings
total_matings_B = sum(matselB)

# Calculate the sum of coancestry values for all mated pairs
sum_coancestry_B = sum(matselB .* H)

# Calculate the average future coancestry
avg_coancestry_B = sum_coancestry_B / total_matings_B

# Get BVs and number of matings for dams and sires with matings
uhat_d_mated_B = uhat[damselindG][nmatdamB .> 0]
nmat_d_mated_B = nmatdamB[nmatdamB .> 0]
uhat_s_mated_B = uhat[sireselindG][nmatsireB .> 0]
nmat_s_mated_B = nmatsireB[nmatsireB .> 0]

# Calculate the mean BV of the mated parents
sum_uhat_parents_B = sum(uhat_d_mated_B .* nmat_d_mated_B) + sum(uhat_s_mated_B .* nmat_s_mated_B)
mean_uhat_parents_B = sum_uhat_parents_B / (2 * total_matings_B)

# Calculate the population mean BV
mean_uhat_population = mean(uhat)

# Calculate the future genetic gain
genetic_gain_B = mean_uhat_parents_B - mean_uhat_population

println("Future genetic gain (Binary Matings): ", genetic_gain_B)



###########################################################################################
#Mating allocations using Mixed Interger Quadratic Programming prop to OCS, integer matings
###########################################################################################

mall2 = Model(SCIP.Optimizer)
@variable(mall2, 0 <= MI[1:nd, 1:ns] <=3, Int);
#Constraint max number of matings for the selected females 
@constraint(mall2, fem_mateI[i in 1:nd], sum(MI[i, j] for j in 1:ns) <= kappad);
#Constraint min number of matings for the selected males 
@constraint(mall2, male_mateI[j in 1:ns], sum(MI[i, j] for i in 1:nd) >= kappas);
@objective(mall2, Min, c_d'*(H .* MI)*c_s); #Minimize coancestry of selected matings

optimize!(mall2)
assert_is_solved_and_feasible(mall2)
println("Optimal value: ", objective_value(mall2))

matselI = round.(Int,(value.(MI)))
nmatdamI = round.(Int,(sum.(eachrow(matselI)))) #Number of matings for each female
nmatsireI = round.(Int,(sum.(eachcol(matselI)))) #Number of matings for each male
siremanzeroI = nmatsireI[nmatsireI.==0] #Sires with zero matings
siremaebvzeroI = uhat[sireselindG][nmatsireI.==0] #EBVs of sires with zero matings
siremaindI = sireselindG[nmatsireI.>0] #Original indices of sires with matings > 0
siremanI = nmatsireI[nmatsireI.>0] #Number of matings of sires with matings > 0
siremaebvI = uhat[sireselindG][nmatsireI.>0] #TBVs of sires with matings > 0
#print([siremaindI,siremanI,sirematbvI])

MAindnozeroI = findall(!iszero, matselI) #Extract non-zero matings

#Make a dictionary of the extracted matings
MAindI = IdDict()
for i in 1:nd
    for j in 1:ns
        MAindI[i,j] = [damselindG[i],sireselindG[j]]
    end
end

#Print out the matings with original id and number of matings[[dam,sire],number of matings]
for i in 1:size(MAindnozeroI)[1]
  println([MAindI[MAindnozeroI[i][1],MAindnozeroI[i][2]],matselI[MAindnozeroI[i][1],MAindnozeroI[i][2]]])
end

#Print out the matings to a text file
open("C://Users//patwa//Documents//QTLMAS2010MA1I.txt", "w") do io
    # Write the column headers
    @printf(io, "%s\t%s\t%s\n", "Dam", "Sire", "Nmat")

    # Loop to write the data
    for i in 1:size(MAindnozeroI)[1]
        matingsI = MAindI[MAindnozeroI[i][1], MAindnozeroI[i][2]]
        nmatselI = matselI[MAindnozeroI[i][1], MAindnozeroI[i][2]]
        
        # Write the data for each row
        @printf(io, "%d\t%d\t%d\n", matingsI[1], matingsI[2], nmatselI)
    end
end


#Plot the number of matings against BVs for the integer setting
scatter(siremaebvzeroI,siremanzeroI,label="Sire zero matings")
xlabel!("BV")
ylabel!("Number of matings")
scatter!(uhat[damselindG],nmatdamI,label="Dam matings")
scatter!(siremaebvI,siremanI,label="Sire matings")
savefig("C://Users//patwa//Documents//QTLMAS2010MA1I.png")

# Calculate the total number of matings
total_matings_I = sum(matselI)

# Calculate the sum of coancestry values for all mated pairs
sum_coancestry_I = sum(matselI .* H)

# Calculate the average future coancestry
avg_coancestry_I = sum_coancestry_I / total_matings_I


# Get BVs and number of matings for dams and sires with matings
uhat_d_mated_I = uhat[damselindG][nmatdamI .> 0]
nmat_d_mated_I = nmatdamI[nmatdamI .> 0]
uhat_s_mated_I = uhat[sireselindG][nmatsireI .> 0]
nmat_s_mated_I = nmatsireI[nmatsireI .> 0]

# Calculate the mean BV of the mated parents
sum_uhat_parents_I = sum(uhat_d_mated_I .* nmat_d_mated_I) + sum(uhat_s_mated_I .* nmat_s_mated_I)
mean_uhat_parents_I = sum_uhat_parents_I / (2 * total_matings_I)

# Calculate the future genetic gain
genetic_gain_I = mean_uhat_parents_I - mean_uhat_population

println("Future genetic gain (Integer Matings): ", genetic_gain_I)




##################################################################################################
##Comparison of GOCSMA with random mating between OC selected individuals and truncation selection
##################################################################################################

# Set up the simulation parameters
n_sim = 100 # Number of random mating simulations
nd = size(damselindG)[1]
ns = size(sireselindG)[1]

#-------------------------------------------------------------------------------
# Random Mating Model among OC selected individuals
#-------------------------------------------------------------------------------

println("\nSimulating random matings among OC selected individuals")

random_coancestry = zeros(n_sim)
random_parent_ebv = zeros(n_sim)
total_matings = sum(matselB)

for sim in 1:n_sim
    # Initialize for this simulation
    dam_mating_count = zeros(Int, nd)
    sim_uhat_parents = zeros(2 * total_matings_B)
    sim_coancestry = zeros(total_matings_B)
    
    mating_count = 0
    while mating_count < total_matings_B
        # Randomly select a dam, ensuring max matings constraint
        chosen_dam_idx = rand(1:nd)
        if dam_mating_count[chosen_dam_idx] < kappad
            # Randomly select a sire
            chosen_sire_idx = rand(1:ns)
            
            # Store parent EBVs
            sim_uhat_parents[2*mating_count + 1] = uhat[damselindG[chosen_dam_idx]]
            sim_uhat_parents[2*mating_count + 2] = uhat[sireselindG[chosen_sire_idx]]
            
            # Store coancestry
            sim_coancestry[mating_count + 1] = H[chosen_dam_idx, chosen_sire_idx]
            
            # Update counters
            dam_mating_count[chosen_dam_idx] += 1
            mating_count += 1
        end
    end
    
    # Calculate and store the average for this simulation
    random_coancestry[sim] = mean(sim_coancestry)
    random_parent_ebv[sim] = mean(sim_uhat_parents)
end

# Final averages from random mating simulations
mean_random_coancestry = mean(random_coancestry)
mean_random_ebv = mean(random_parent_ebv)
genetic_gain_random = mean_random_ebv - mean(uhat)

println("Random Mating (Binary): Avg Coancestry = ", mean_random_coancestry)
println("Random Mating (Binary): Genetic Gain = ", genetic_gain_random)


#-------------------------------------------------------------------------------
# Random Mating Model among top ranked individuals
#-------------------------------------------------------------------------------
# Sort individuals by BV within each sex
sires_sorted_indices = sortperm(uhat[s], rev=true)
dams_sorted_indices = sortperm(uhat[d], rev=true)

# Get the original indices of the selected sires and dams
top_sires_indices = findall(s)[sires_sorted_indices[1:ns]]
top_dams_indices = findall(d)[dams_sorted_indices[1:nd]]

# Extract BVs and the H-matrix for the top-selected individuals
uhat_top_sires = uhat[top_sires_indices]
uhat_top_dams = uhat[top_dams_indices]

# Extract the H matrix (reduced G matrix) of the top-selected individuals
H_top = zeros(nd, ns)
for i in 1:nd
    for j in 1:ns
        H_top[i, j] = G[top_dams_indices[i], top_sires_indices[j]]
    end
end

## Comparison with top-ranked random mating
println("\nSimulating random matings for top-ranked individuals...")

# Use the same number of simulations and total matings as the other models
n_sim = 100
total_matings_B = sum(matselB) # Use total matings from binary OC model for consistency

random_coancestry_top = zeros(n_sim)
random_parent_ebv_top = zeros(n_sim)

for sim in 1:n_sim
    # Initialize for this simulation
    dam_mating_count = zeros(Int, nd)
    sim_uhat_parents = zeros(2 * total_matings_B)
    sim_coancestry = zeros(total_matings_B)
    
    mating_count = 0
    while mating_count < total_matings_B
        # Randomly select a dam, ensuring max matings constraint
        chosen_dam_idx = rand(1:nd)
        if dam_mating_count[chosen_dam_idx] < kappad
            # Randomly select a sire
            chosen_sire_idx = rand(1:ns)
            
            # Store parent EBVs
            sim_uhat_parents[2*mating_count + 1] = uhat_top_dams[chosen_dam_idx]
            sim_uhat_parents[2*mating_count + 2] = uhat_top_sires[chosen_sire_idx]
            
            # Store coancestry from the H_top matrix
            sim_coancestry[mating_count + 1] = H_top[chosen_dam_idx, chosen_sire_idx]
            
            # Update counters
            dam_mating_count[chosen_dam_idx] += 1
            mating_count += 1
        end
    end
    
    # Calculate and store the average for this simulation
    random_coancestry_top[sim] = mean(sim_coancestry)
    random_parent_ebv_top[sim] = mean(sim_uhat_parents)
end

# Final averages from random mating simulations
mean_random_coancestry_top = mean(random_coancestry_top)
mean_random_ebv_top = mean(random_parent_ebv_top)
genetic_gain_random_top = mean_random_ebv_top - mean(uhat)

println("Top-Ranked Mating: Avg Coancestry = ", mean_random_coancestry_top)
println("Top-Ranked Mating: Genetic Gain = ", genetic_gain_random_top)
