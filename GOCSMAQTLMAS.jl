# Read QTLMAS2010 data
using DelimitedFiles, Statistics
tbvdata = readdlm("C://Users//pwaldman20//Documents//tbv.txt",',')
gender = tbvdata[:,2]
sire = gender .==1 #Male/Sire indicator variable
dam = gender .==0 #Female/Dam indicator variable
tbv = tbvdata[:,5] #True Breeding Values
X = readdlm("C://Users//pwaldman20//Documents//QTLMAS2010gen.txt",',') #Markers (SNPs in 0,1,2 format)

n = size(X)[1] #Number of individuals

#G-matrix VanRaden1
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

#OCS model with G
using JuMP, LinearAlgebra, COSMO
mG = JuMP.Model(COSMO.Optimizer);
theta = 0.001 #The allowed level of coancestry/inbreeding
@variable(mG, oc[1:n] >= 0) #The optimum contribution variable with positivity constraint
@objective(mG, Max, dot(tbv,oc)); #The objective finction to be maximized
@constraint(mG, sum(dam[i]*oc[i] for i in 1:n) == 0.5); #The sum constraint on females/dams
@constraint(mG, sum(sire[i]*oc[i] for i in 1:n) == 0.5); #The sum constraint on males/sires
@constraints(mG, begin
  sum(oc[i] * G[i, j] * oc[j] for i = 1:n, j = 1:n)/2 <= theta
end) #The quadratic constraint

#Run the model
JuMP.optimize!(mG);

#Extract OC values from JuMP
contrG = JuMP.value.(oc)

#Set small values to zero and extract important CO individuals
selcontrind = contrG.>0.0001
cosel = contrG[selcontrind]
tbvsel = tbv[selcontrind]
indseq = 1:n
selind = indseq[selcontrind]
nsel = size(selind)[1]
damselindG = selind[dam[selind].==1]
sireselindG = selind[sire[selind].==1]
nd = size(damselindG)[1]
ns = size(sireselindG)[1]

#Plot the OC values against the TBVs with separate lables
using Plots
scatter(tbv,contrG,label="No OC")
xlabel!("TBV")
ylabel!("OC")
scatter!(tbv[damselindG],contrG[damselindG],label="Dam OC")
scatter!(tbv[sireselindG],contrG[sireselindG],label="Sire OC")
savefig("C://Users//pwaldman20//Documents//QTLMAS2010OCG2.png")


#Extract the H matrix (reduced G matrix) of the selected individuals
H = zeros(nd, ns)
for i in 1:nd
    for j in 1:ns
        H[i, j] = G[damselindG[i],sireselindG[j]]
    end
end

#Mating allocations using Mixed Interger Quadratic Programming prop to OCS, binary matings
kappad = 3 #Max number of matings per female
c_s = contrG[sireselindG] #Extract the OC values for selected sires
c_d = contrG[damselindG] #Extract the OC values for selected dams
using SCIP
mall1 = Model(SCIP.Optimizer)
@variable(mall1, MB[1:nd, 1:ns], Bin); #Variable allowing only binary (0/1) matings
#Constraint max number of matings for the selected females 
@constraint(mall1, fem_mateB[i in 1:nd], sum(MB[i, j] for j in 1:ns) <= kappad);
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
sirematbvzeroB = tbv[sireselindG][nmatsireB.==0] #TBVs of sires with zero matings
siremaindB = sireselindG[nmatsireB.>0] #Original indices of sires with matings > 0
siremanB = nmatsireB[nmatsireB.>0] #Number of matings of sires with matings > 0
sirematbvB = tbv[sireselindG][nmatsireB.>0] #TBVs of sires with matings > 0
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
  print([MAindB[MAindnozeroB[i][1],MAindnozeroB[i][2]],matselB[MAindnozeroB[i][1],MAindnozeroB[i][2]]])
end


#Plot the number of matings against TBVs for the binary setting
scatter(sirematbvzeroB,siremanzeroB,label="Sire zero matings")
xlabel!("TBV")
ylabel!("Number of matings")
scatter!(tbv[damselindG],nmatdamB,label="Dam matings")
scatter!(sirematbvB,siremanB,label="Sire matings")
savefig("C://Users//pwaldman20//Documents//QTLMAS2010MA1B.png")


#Mating allocations using Mixed Interger Quadratic Programming prop to OCS, integer matings
mall2 = Model(SCIP.Optimizer)
@variable(mall2, 0 <= MI[1:nd, 1:ns] <=3, Int);
#Constraint max number of matings for the selected females 
@constraint(mall2, fem_mateI[i in 1:nd], sum(MI[i, j] for j in 1:ns) <= kappad);
@objective(mall2, Min, c_d'*(H .* MI)*c_s); #Minimize coancestry of selected matings

optimize!(mall2)
assert_is_solved_and_feasible(mall2)
println("Optimal value: ", objective_value(mall2))

matselI = round.(Int,(value.(MI)))
nmatdamI = round.(Int,(sum.(eachrow(matselI)))) #Number of matings for each female
nmatsireI = round.(Int,(sum.(eachcol(matselI)))) #Number of matings for each male
siremanzeroI = nmatsireI[nmatsireI.==0] #Sires with zero matings
sirematbvzeroI = tbv[sireselindG][nmatsireI.==0] #TBVs of sires with zero matings
siremaindI = sireselindG[nmatsireI.>0] #Original indices of sires with matings > 0
siremanI = nmatsireI[nmatsireI.>0] #Number of matings of sires with matings > 0
sirematbvI = tbv[sireselindG][nmatsireI.>0] #TBVs of sires with matings > 0
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
  print([MAindI[MAindnozeroI[i][1],MAindnozeroI[i][2]],matselI[MAindnozeroI[i][1],MAindnozeroI[i][2]]])
end


#Plot the number of matings against TBVs for the integer setting
scatter(sirematbvzeroI,siremanzeroI,label="Sire zero matings")
xlabel!("TBV")
ylabel!("Number of matings")
scatter!(tbv[damselindG],nmatdamI,label="Dam matings")
scatter!(sirematbvI,siremanI,label="Sire matings")
savefig("C://Users//pwaldman20//Documents//QTLMAS2010MA1I.png")
