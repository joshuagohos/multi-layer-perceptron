#*****************************************************************
# PROJECT 2: BACKPROPAGATION - by Josh Goh 20 Mar 2007
#*****************************************************************

# Set project parameters
runs   = 10 		# No. of runs per condition
err    = 0.001		# Network settling error criterion
epochs = 100000	        # No. of epochs to train if no settling
eta    = 0.3    	# Growth rate
mu     = 0.7    	# Momentum


# Specify output file paths
# Parameter reports
parrepfilepath=sprintf("%s/bppar_eta%0.2f_mu%0.2f_jg.txt",getwd(),eta,mu)
parrepfile<-file(parrepfilepath,"a")

# Simulation reports
outputfilepath=sprintf("%s/bpout_eta%0.2f_mu%0.2f_jg.txt",getwd(),eta,mu)
outputfile<-file(outputfilepath,"a")


#-----------------------------------------------------------------
# READ FUNCTIONS INTO WORKSPACE
#-----------------------------------------------------------------


# Function 1: trainnet_perceptron
#-----------------------------------------------------------------
# Trains a network on desired input and output mappings using the delta
# learning rule with specified growth rate and momentum. Input activation is 
# propagated through the network each iteration using the logistic activation 
# function. The actual output is compared to the desired output. The error 
# indicates how much the connection weights through each layer should change 
# using the generalized delta rule. These weight changes are backpropagated 
# through each layer.
#
# Returns a list of weights, Wt, over all training epochs (decided by user),
# and total network error, toterr, over epochs.
#
# Usage:
#
# trainnet_perceptron(W,input,dout,unitslayer,epochs,err,eta,mu)
#
# W           - List of connection weight matrices for each layer of connections.
#               This is initialized as a matrix of random weights for each 
#               layer if set as NULL.
# input       - Matrix of input patterns (rows: units; columns: patterns)
# dout        - Matrix of desired output pattern (rows: units; columns: patterns)
# unitslayers - Vector of number of units in each layer of units
# epochs      - Number of cycles before network stops training
# err         - Error threshold below which to stop training
# eta         - Growth rate, blame
# mu          - Momentum

trainnet_perceptron<-function(W,input,dout,unitslayer,epochs,err,eta,mu) {
	
	# SET PARAMETERS
	ninput    = dim(input)[1]         	# No. of input units
	npats     = dim(input)[2]         	# No. of training patterns
	nlayers   = length(unitslayer)-1  	# No. of connection layers
	Wt        = list()		  			# Weights over epochs
	hWt       = list()		 			# Held weights over epochs
	toterr    = c()			  			# Total network error
	toterr[1] = err+10		  			# Set toterr > err by arbitrary amount
	
	# Add bias units
	for (i in 1:nlayers) {
		unitslayer[i]=unitslayer[i]+1
	}

	# INITIALIZE W AND HOLD W MATRICES
	if (is.null(W)) {

		W            = list()
		holdpatternW = list()

		# Initialize connection layers
		for (k in 1:nlayers) {				
			hW  = matrix(runif(unitslayer[k]*unitslayer[k+1],-1,1),unitslayer[k],unitslayer[k+1])
			hhW = matrix(0,unitslayer[k],unitslayer[k+1])
			
			# Remove connection to next layer bias unit
			if (k!=nlayers) {
				hW[,unitslayer[k+1]]=0
				hhW[,unitslayer[k+1]]=0
			}
						
			W[[k]]            = hW
			holdpatternW[[k]] = hhW				
		}
	}
	else {
	 	holdpatternW = c()

		# Initialize connection layers
		for (k in 1:nlayers) {
			hhW = matrix(0,unitslayer[k],unitslayer[k+1])
			
			# Check and remove connection to next layer bias unit
			if (k!=nlayers) {
				if (unitslayer[k+1]==dim(as.matrix(W[[k]]))[2]) {
					W[[k]][,unitslayer[k+1]]=0
				}
			}					
			holdpatternW[[k]] = hhW				
		}
	}

	zeroholdpatternW=holdpatternW

	# End W initialization

	# TRAIN W
	for (t in 1:epochs) { 		# Epoch loop
		if (!is.nan(toterr[1])) { # NaN check
		if (toterr[1]>=err) {	# Error loop
			
			# Set parameters
			holdlayerW   = c()
			paterr       = c()
			
			# Momentum term
			M=c()
			for (k in 1:nlayers) {
			M[[k]] = holdpatternW[[k]]*mu
			}
			
			holdpatternW = zeroholdpatternW
		
			for (p in 1:npats) {

				# Set parameters
				ai      = c()
				ai[[1]] = input[,p]
				error   = c()

				# Propagate activity
				for (k in 1:nlayers) {
			
					# Set bias unit activation to 1
					ai[[k]][unitslayer[k]]=1
					
					# Logistic activation function
					ai[[k+1]]=1/(1+(exp(-(t(W[[k]])%*%ai[[k]]))))

				}

				# Calculate change in weights
				for (k in nlayers:1) {

					# For output layer
					if (k==nlayers) {

						# Calculate output unit error
						error[[k+1]]=(dout[,p]-ai[[k+1]])*ai[[k+1]]*(1-ai[[k+1]])
					
						# Calculate pattern error
						paterr[p]=sum((dout[,p]-ai[[k+1]])^2)
						
						# Calculate layer weight change
						holdlayerW[[k]]=(eta*ai[[k]]%*%t(error[[k+1]]))
					}

					# For non-output layers
					else {

						# Calculate non-output unit error
						error[[k+1]]=(W[[k+1]]%*%error[[k+2]])*ai[[k+1]]*(1-ai[[k+1]])
						
						# Calculate layer weight change
						holdlayerW[[k]]=(eta*(ai[[k]]%*%t(error[[k+1]])))
						
						# Account for bias unit
						holdlayerW[[k]][,unitslayer[k+1]]=0
						
					}
				} # End layer loop
				
				# Sum pattern dW
				for (l in 1:length(holdlayerW)) {
					holdpatternW[[l]]=holdlayerW[[l]]+holdpatternW[[l]]
				}
				
			} # End pattern loop

			# Batch add dW to original weights
			for (l in 1:length(W)) {
				W[[l]]=W[[l]]+holdpatternW[[l]]+M[[l]]
			}
		
			# Calculate total error
			toterr[1]=sum(paterr)
	
			# Update variables over epochs
			Wt[[1]]=W
		
		} # End error IF loop
		
		else {break}
		
		} # End NaN check
		
		else {break} 
		
	} # End epoch loop

	# Assign variable to global environment
	assign("Wt",Wt,envir=.GlobalEnv)
	assign("toterr",toterr,envir=.GlobalEnv)
	assign("tepoch",t,envir=.GlobalEnv)

} # End function 1


# Function 2: testnet_perceptron
#-----------------------------------------------------------------
# Tests a trained perceptron network, W, for the output, a, given
# an input. Each unit state is determined using a logistic activation
# function. Returns a, a vector of activation states for each layer
# of units including the input layer.
#
# Usage:
#
# testnet_perceptron(W,input,unitslayer)
#
# W          - Trained connection weight matrix, must be list format
#	           (see trainnet_perceptron.R)
# input      - Test pattern vector (units: rows)
# unitslayer - Vector of number of units in each layer of units

testnet_perceptron=function(W,input,unitslayer) {

	# SET PARAMETERS
	nlayers = length(W)	# No. of connection layers
	a	= c()
	a[[1]]  = input
	ar	= c()
	ar[[1]] = input
	
	# Add bias units
	for (i in 1:nlayers) {
		unitslayer[i]=unitslayer[i]+1
	}
	
	# TEST PATTERN	
	
	for (k in 1:nlayers) {
		atemp=c()
		for (j in 1:unitslayer[k+1]) {
				
			# Set bias unit activation to 1
			a[[k]][unitslayer[k]]=1
		
			# Logistic activation function
			atemp[j]=1/(1+(exp(1)^(-t(W[[k]][1:unitslayer[k],j])%*%a[[k]][1:unitslayer[k]])))
			
		}
		a[[k+1]]=atemp
		ar[[k+1]]=round(atemp)
	}
	
	# Assign variable to global environment
	assign("a",a,envir=.GlobalEnv)
	assign("ar",ar,envir=.GlobalEnv)
	
} # End function 2

# End read functions


#-----------------------------------------------------------------
# BEGIN WORKSPACE COMPUTATIONS FOR PROJECT SIMULATIONS
#-----------------------------------------------------------------



# PART 1: CATEGORY LEARNING
#-----------------------------------------------------------------

# Create list of backprop networks
bnet      = c()
bnet[[1]] = c(10,4)	# 1 layer, 10 input, 0  hidden, 4 output
bnet[[2]] = c(10,2,4)	# 2 layer, 10 input, 2  hidden, 4 output
bnet[[3]] = c(10,3,4)	# 2 layer, 10 input, 3  hidden, 4 output
bnet[[4]] = c(10,4,4)	# 2 layer, 10 input, 4  hidden, 4 output
bnet[[5]] = c(10,7,4)	# 2 layer, 10 input, 7  hidden, 4 output
bnet[[6]] = c(10,10,4)	# 2 layer, 10 input, 10 hidden, 4 output
bnet[[7]] = c(10,15,4)	# 2 layer, 10 input, 15 hidden, 4 output
bnet[[8]] = c(10,20,4)	# 2 layer, 10 input, 20 hidden, 4 output

#Print network types to parameter report file
cat("PART 1: Network type units and layers, input(left)-hidden(middle)-output(right)\n",file=parrepfilepath,append=TRUE)
for (net in 1:length(bnet)) {
	cat(sprintf("Network type %.0f:\t",net),file=parrepfilepath,append=TRUE)
	cat(bnet[[net]],file=parrepfilepath,append=TRUE)
	cat("\n",file=parrepfilepath,append=TRUE)
}
cat("\n",file=parrepfilepath,append=TRUE)
cat("PART 1: Test patterns, rows(units), columns(runs)\n",file=parrepfilepath,append=TRUE)


# Create training input and outputs
# Set parameters
input  = c()
output = c()
	
# Set1: non-overlapping categories
# Inputs
iset1 =             matrix(c(0,0,0,0,0,1,0,0,0,0),10,1)  # Cat A
iset1 = cbind(iset1,matrix(c(0,0,0,0,0,0,0,0,0,1),10,1)) # Cat A
iset1 = cbind(iset1,matrix(c(0,0,0,0,1,0,0,0,1,0),10,1)) # Cat B
iset1 = cbind(iset1,matrix(c(0,0,0,1,0,0,0,0,0,0),10,1)) # Cat B
iset1 = cbind(iset1,matrix(c(0,0,0,0,0,0,0,1,0,0),10,1)) # Cat C
iset1 = cbind(iset1,matrix(c(0,0,1,0,0,0,0,0,0,0),10,1)) # Cat C
iset1 = cbind(iset1,matrix(c(0,1,0,0,0,0,1,0,0,0),10,1)) # Cat D
iset1 = cbind(iset1,matrix(c(1,0,0,0,0,0,1,0,0,0),10,1)) # Cat D
		
input[[1]] = iset1

# Desired outputs
oset1 =             matrix(c(1,0,0,0),4,1)  # Cat A
oset1 = cbind(oset1,matrix(c(1,0,0,0),4,1)) # Cat A
oset1 = cbind(oset1,matrix(c(0,1,0,0),4,1)) # Cat B
oset1 = cbind(oset1,matrix(c(0,1,0,0),4,1)) # Cat B
oset1 = cbind(oset1,matrix(c(0,0,1,0),4,1)) # Cat C
oset1 = cbind(oset1,matrix(c(0,0,1,0),4,1)) # Cat C
oset1 = cbind(oset1,matrix(c(0,0,0,1),4,1)) # Cat D
oset1 = cbind(oset1,matrix(c(0,0,0,1),4,1)) # Cat D

output[[1]] = oset1

# Set2: linearly independent instances and linearly separable categories
# Inputs
iset2 =             matrix(c(1,1,0,1,0,0,0,0,0,1),10,1)  # Cat A
iset2 = cbind(iset2,matrix(c(1,1,0,0,1,0,0,0,0,1),10,1)) # Cat A
iset2 = cbind(iset2,matrix(c(1,1,0,0,0,1,0,0,1,0),10,1)) # Cat B
iset2 = cbind(iset2,matrix(c(1,1,0,0,0,0,0,0,1,0),10,1)) # Cat B
iset2 = cbind(iset2,matrix(c(1,0,1,1,0,0,0,1,0,0),10,1)) # Cat C
iset2 = cbind(iset2,matrix(c(1,0,1,0,1,0,0,1,0,0),10,1)) # Cat C
iset2 = cbind(iset2,matrix(c(1,0,1,0,0,1,1,0,0,0),10,1)) # Cat D
iset2 = cbind(iset2,matrix(c(1,0,1,0,0,0,1,0,0,0),10,1)) # Cat D
		
input[[2]] = iset2

# Desired outputs (same as set 1)
oset2 = oset1
		
output[[2]] = oset2

# Set3: not linearly separable categories
# Inputs
iset3 =             matrix(c(1,1,1,1,1,0,0,0,0,0),10,1)
iset3 = cbind(iset3,matrix(c(0,0,0,0,0,1,1,1,1,1),10,1))
iset3 = cbind(iset3,matrix(c(1,1,0,1,1,0,0,1,0,0),10,1))
iset3 = cbind(iset3,matrix(c(0,0,1,0,0,1,1,0,1,1),10,1))
		
input[[3]] = iset3

# Desired outputs
oset3 =             matrix(c(1,1,0,0),4,1)
oset3 = cbind(oset3,matrix(c(1,1,0,0),4,1))
oset3 = cbind(oset3,matrix(c(0,0,1,1),4,1))
oset3 = cbind(oset3,matrix(c(0,0,1,1),4,1))
		
output[[3]] = oset3

# Clear unecessary variables
rm(iset1,iset2,iset3,oset1,oset2,oset3)

# Create distorted test patterns
# Set parameters
prob<-c(0,0.3,0.5) # List distortion probabilities
testpp=c()

for (s in 1:length(input)) { # set counter

	# Initiate a zero test pattern matrix with activation placeholders for
	# each unit, training pattern, distortion probability, and run
	testp<-c(rep(0,(dim(input[[s]])[1]*dim(input[[s]])[2]*length(prob)*runs)))
	dim(testp)<-c(dim(input[[s]])[1],dim(input[[s]])[2],length(prob),runs)

	# Fill in testp
	for (p in 1:length(prob)) {                                       # Distortion probability counter
		pdist<-c(rep(1,(100*prob[p])),rep(0,(100-(100*prob[p])))) # Create probability distribution
		for (pat in 1:dim(input[[s]])[2]) {                       # Training pattern counter
			for (patv in 1:dim(input[[s]])[1]) {              # Unit counter
				for (nrun in 1:runs) {                    # Run counter
					pn<-sample(pdist,1)               # Randomly select from distribution

					# Update test activation state
					if (pn==1) {
						if (input[[s]][patv,pat]==1) {
							testp[patv,pat,p,nrun]<-0
						}
						else {
							testp[patv,pat,p,nrun]<-1
						}
					}
					else {
						testp[patv,pat,p,nrun]<-input[[s]][patv,pat]
					} # End update state
				}     # End run counter
			}         # End unit counter

		#Print test pattern to parameter report file
		cat(sprintf("PART 1 SET %.0f: Training pattern %.0f with distortion probability: %.1f\n",s,pat,prob[p]),file=parrepfilepath,append=TRUE)
		write.table(testp[,pat,p,],sep="\t",file=parrepfilepath,append=TRUE,col.names=FALSE)
		cat("\n",file=parrepfilepath,append=TRUE)

		} 	# End training pattern counter
	} 		# End distortion probability counter
	testpp[[s]]=testp
} # End set counter

# Initiate list of report variables for each network
bnepoch = c() # No. of epochs to settle
bferr   = c() # Error between final and desired output per network
sferr   = c() # Error between final and desired output per set
srpfa   = c() # Final test activation state per set
bsrpfa  = c() # Final test activation state per network

# Print final activation states to parameter report file
cat("PART 1: Final test activation states (rows: output units; cols: runs)\n",file=parrepfilepath,append=TRUE)

# Perform training and tests
for (b in 1:length(bnet)) {	# Network counter
	
	# Initiate matrix no. of epochs to settle per set per run
	nepochs      = c(rep(0,length(input)*runs))
	dim(nepochs) = c(length(input),runs)
	
	for (s in 1:length(input)) {	# Set counter
	
		testp = testpp[[s]]
		
		# Initiate report variables: Error between final output and desired output
		ferr      = c(rep(0,dim(testp)[2]*dim(testp)[3]*dim(testp)[4]))
		dim(ferr) = c(dim(testp)[2],dim(testp)[3],dim(testp)[4])
		rpfa      = c()

		# Train and test network
		for (r in 1:dim(testp)[4]) {
		
			# Train
			trainnet_perceptron(NULL,input[[s]],output[[s]],bnet[[b]],epochs,err,eta,mu)
			
			# Store nepochs
			nepochs[s,r]=tepoch
			
			pfa=c()

			# Test
			for (p in 1:dim(testp)[3]) {
				fa=c()
				for (i in 1:dim(testp)[2]) {
					testnet_perceptron(Wt[[length(Wt)]],as.matrix(testp[,i,p,r]),bnet[[b]])
					
					# Print rounded final test activation state
					fa=cbind(fa,ar[[length(ar)]])
					cat(sprintf("Net %.0f, Set %.0f, Distortion Prob %.1f, Pattern %.0f, Run %.0f\n",b,s,prob[p],i,r),file=parrepfilepath,append=TRUE)
					for (la in 1:length(ar)) {
						cat(ar[[la]],file=parrepfilepath,append=TRUE)
						cat("\n",file=parrepfilepath,append=TRUE)
					}
					cat("\n",file=parrepfilepath,append=TRUE)
					
					# Calculate ferr from actual final test activation state
					d             = as.matrix(output[[s]][,i])-as.matrix(a[[length(a)]])
					ferr[i,p,r]   = sum(d^2)
					
				}
				pfa[[p]]=fa
			}
			rpfa[[r]]=pfa
		}
		
		srpfa[[s]] = rpfa
		sferr[[s]] = ferr

	} # End set counter
	
	# Update report variables
	bnepoch[[b]] = nepochs
	bferr[[b]]   = sferr
	bsrpfa[[b]]  = srpfa
	
} # End network counter



#Print reports
cat("PART 1: CATEGORY LEARNING\n",file=outputfilepath,append=TRUE)
for (set in 1:length(input)) {
	cat(sprintf("Set %.0f: Mean (SD) number of epochs required to reach error threshold for each network type\n",set),file=outputfilepath,append=TRUE)
	for (net in 1:length(bnepoch)) {
		cat(mean(bnepoch[[net]][set,]),file=outputfilepath,append=TRUE)
		cat("\t(",file=outputfilepath,append=TRUE)
		cat(sd(bnepoch[[net]][set,]),file=outputfilepath,append=TRUE)
		cat(")\n",file=outputfilepath,append=TRUE)
	}
	cat("\n",file=outputfilepath,append=TRUE)
}

for (set in 1:length(input)) {
	cat(sprintf("Set %.0f: Mean (SD) error between desired and actual output for each network type\n",set),file=outputfilepath,append=TRUE)
	for (net in 1:length(bnepoch)) {
		cat(sprintf("Network type %.0f\n",net),file=outputfilepath,append=TRUE)
		for (pn in 1:length(prob)) {
			cat(sprintf("For distortion probability %.1f (rows: training pattern)\n",prob[pn]),file=outputfilepath,append=TRUE)
			for (pat in 1:dim(as.matrix(input[[set]]))[2]) {
				cat(mean(bferr[[net]][[set]][pat,pn,]),file=outputfilepath,append=TRUE)
				cat("\t(",file=outputfilepath,append=TRUE)
				cat(sd(bferr[[net]][[set]][pat,pn,]),file=outputfilepath,append=TRUE)
				cat(")\n",file=outputfilepath,append=TRUE)
			}
			cat("\n",file=outputfilepath,append=TRUE)
		}
	}
}



# PART 2A: DATA COMPRESSION
#-----------------------------------------------------------------
# Create backprop networks
bnet      = c()
bnet[[1]] = c(8,3,8)	# 1 layer, 8 input, 3  hidden, 8 output

#Print network types to parameter report file
cat("PART 2A: Network type units and layers, input(left)-hidden(middle)-output(right)\n",file=parrepfilepath,append=TRUE)
for (net in 1:length(bnet)) {
	cat(sprintf("Network type %.0f:\t",net),file=parrepfilepath,append=TRUE)
	cat(bnet[[net]],file=parrepfilepath,append=TRUE)
	cat("\n",file=parrepfilepath,append=TRUE)
}
cat("\n",file=parrepfilepath,append=TRUE)
cat("PART 2A: Test patterns, rows(units), columns(runs)\n",file=parrepfilepath,append=TRUE)


# Create training input and outputs
# Set parameters
train  = c()
input  = c()
output = c()
	
# Set1
# Inputs
iset1 =             matrix(c(1,0,0,0,0,0,0,0),8,1)  # Cat 1
iset1 = cbind(iset1,matrix(c(0,1,0,0,0,0,0,0),8,1)) # Cat 2
iset1 = cbind(iset1,matrix(c(0,0,1,0,0,0,0,0),8,1)) # Cat 3
iset1 = cbind(iset1,matrix(c(0,0,0,1,0,0,0,0),8,1)) # Cat 4
iset1 = cbind(iset1,matrix(c(0,0,0,0,1,0,0,0),8,1)) # Cat 5
iset1 = cbind(iset1,matrix(c(0,0,0,0,0,1,0,0),8,1)) # Cat 6
iset1 = cbind(iset1,matrix(c(0,0,0,0,0,0,1,0),8,1)) # Cat 7
		
train[[1]] = iset1

iset1 = cbind(iset1,matrix(c(0,0,0,0,0,0,0,1),8,1)) # Cat 8

input[[1]] = iset1

# Desired outputs
output[[1]] = input[[1]]

# Clear unecessary variables
rm(iset1)

# Create distorted test patterns
# Set parameters
prob<-c(0) # List distortion probabilities
testpp=c()

for (s in 1:length(input)) { # set counter

	# Initiate a zero test pattern matrix with activation placeholders for
	# each unit, training pattern, distortion probability, and run
	testp<-c(rep(0,(dim(input[[s]])[1]*dim(input[[s]])[2]*length(prob)*runs)))
	dim(testp)<-c(dim(input[[s]])[1],dim(input[[s]])[2],length(prob),runs)

	# Fill in testp
	for (p in 1:length(prob)) {                                       # Distortion probability counter
		pdist<-c(rep(1,(100*prob[p])),rep(0,(100-(100*prob[p])))) # Create probability distribution
		for (pat in 1:dim(input[[s]])[2]) {                       # Training pattern counter
			for (patv in 1:dim(input[[s]])[1]) {              # Unit counter
				for (nrun in 1:runs) {                    # Run counter
					pn<-sample(pdist,1)               # Randomly select from distribution

					# Update test activation state
					if (pn==1) {
						if (input[[s]][patv,pat]==1) {
							testp[patv,pat,p,nrun]<-0
						}
						else {
							testp[patv,pat,p,nrun]<-1
						}
					}
					else {
						testp[patv,pat,p,nrun]<-input[[s]][patv,pat]
					} # End update state
				}     # End run counter
			}         # End unit counter

		#Print test pattern to parameter report file
		cat(sprintf("PART 2A SET %.0f: Pattern %.0f with distortion probability: %.1f\n",s,pat,prob[p]),file=parrepfilepath,append=TRUE)
		write.table(testp[,pat,p,],sep="\t",file=parrepfilepath,append=TRUE,col.names=FALSE)
		cat("\n",file=parrepfilepath,append=TRUE)

		} 	# End training pattern counter
	} 		# End distortion probability counter
	testpp[[s]]=testp
} # End set counter

# Initiate list of report variables for each network
bnepoch = c() # No. of epochs to settle
bferr   = c() # Error between final and desired output per network
sferr   = c() # Error between final and desired output per set
srpfa   = c() # Final test activation state per set
bsrpfa  = c() # Final test activation state per network

cat("PART 2A: Final test activation states (rows: output units; cols: runs)\n",file=parrepfilepath,append=TRUE)

# Perform training and tests
for (b in 1:length(bnet)) {	# Network counter
	
	# Initiate matrix no. of epochs to settle per set per run
	nepochs      = c(rep(0,length(input)*runs))
	dim(nepochs) = c(length(input),runs)
	
	for (s in 1:length(input)) {	# Set counter
	
		testp = testpp[[s]]

		# Initiate report variables: Error between final output and desired output
		ferr      = c(rep(0,dim(testp)[2]*dim(testp)[3]*dim(testp)[4]))
		dim(ferr) = c(dim(testp)[2],dim(testp)[3],dim(testp)[4])
		rpfa      = c()
		
		Wtr=c() # Initiate weight storage per run for further testing
		
		# Train and test network
		for (r in 1:dim(testp)[4]) {
		
			# Train
			trainnet_perceptron(NULL,train[[s]],output[[s]],bnet[[b]],epochs,err,eta,mu)
			
			# Store variables
			nepochs[s,r]=tepoch
			Wtr[[r]]=Wt[[length(Wt)]]
			
			pfa=c()

			# Test
			for (p in 1:dim(testp)[3]) {
				fa=c()
				for (i in 1:dim(testp)[2]) {
					testnet_perceptron(Wt[[length(Wt)]],as.matrix(testp[,i,p,r]),bnet[[b]])
					
					# Store final rounded test activation state
					fa=cbind(fa,ar[[length(ar)]])
					cat(sprintf("Net %.0f, Set %.0f, Distortion Prob %.1f, Pattern %.0f, Run %.0f\n",b,s,prob[p],i,r),file=parrepfilepath,append=TRUE)
					for (la in 1:length(ar)) {
						cat(ar[[la]],file=parrepfilepath,append=TRUE)
						cat("\n",file=parrepfilepath,append=TRUE)
					}
					cat("\n",file=parrepfilepath,append=TRUE)
					
					# Calculate ferr from actual final test activation states
					d             = as.matrix(output[[s]][,i])-as.matrix(a[[length(a)]])
					ferr[i,p,r]   = sum(d^2)
					
				}
				pfa[[p]]=fa
			}
			rpfa[[r]]=pfa
		}
		
		srpfa[[s]] = rpfa
		sferr[[s]] = ferr

	} # End set counter
	
	# Update report variables
	bnepoch[[b]] = nepochs
	bferr[[b]]   = sferr
	bsrpfa[[b]]  = srpfa
	
} # End network counter



#Print reports
cat("PART 2A: DATA COMPRESSION\n",file=outputfilepath,append=TRUE)
for (set in 1:length(input)) {
	cat(sprintf("Set %.0f: Mean (SD) number of epochs required to reach error threshold for each network type\n",set),file=outputfilepath,append=TRUE)
	for (net in 1:length(bnepoch)) {
		cat(mean(bnepoch[[net]][set,]),file=outputfilepath,append=TRUE)
		cat("\t(",file=outputfilepath,append=TRUE)
		cat(sd(bnepoch[[net]][set,]),file=outputfilepath,append=TRUE)
		cat(")\n",file=outputfilepath,append=TRUE)
	}
	cat("\n",file=outputfilepath,append=TRUE)
}

for (set in 1:length(input)) {
	cat(sprintf("Set %.0f: Mean (SD) error between desired and actual output for each network type\n",set),file=outputfilepath,append=TRUE)
	for (net in 1:length(bnepoch)) {
		cat(sprintf("Network type %.0f\n",net),file=outputfilepath,append=TRUE)
		for (pn in 1:length(prob)) {
			cat(sprintf("For distortion probability %.1f (rows: training pattern)\n",prob[pn]),file=outputfilepath,append=TRUE)
			for (pat in 1:dim(as.matrix(input[[set]]))[2]) {
				cat(mean(bferr[[net]][[set]][pat,pn,]),file=outputfilepath,append=TRUE)
				cat("\t(",file=outputfilepath,append=TRUE)
				cat(sd(bferr[[net]][[set]][pat,pn,]),file=outputfilepath,append=TRUE)
				cat(")\n",file=outputfilepath,append=TRUE)
			}
			cat("\n",file=outputfilepath,append=TRUE)
		}
	}
}





# PART 2B: GENERALIZATION
#-----------------------------------------------------------------

#Print network types to parameter report file
cat("PART 2B: Network type units and layers, input(left)-hidden(middle)-output(right)\n",file=parrepfilepath,append=TRUE)
for (net in 1:length(bnet)) {
	cat(sprintf("Network type %.0f:\t",net),file=parrepfilepath,append=TRUE)
	cat(bnet[[net]],file=parrepfilepath,append=TRUE)
	cat("\n",file=parrepfilepath,append=TRUE)
}
cat("\n",file=parrepfilepath,append=TRUE)
cat("PART 2B: Test patterns, rows(units), columns(runs)\n",file=parrepfilepath,append=TRUE)


# Create training input and outputs
# Set parameters
train  = c()
input  = c()
output = c()
	
# Set1
# Inputs
iset1 =             matrix(c(1,1,1,0,0,0,0,0),8,1)  # Cat 9
iset1 = cbind(iset1,matrix(c(0,0,0,1,1,1,0,0),8,1)) # Cat 10
		
train[[1]] = iset1

iset1 = cbind(iset1,matrix(c(0,0,1,1,0,0,0,0),8,1)) # Cat 11
iset1 = cbind(iset1,matrix(c(1,0,1,0,0,0,0,0),8,1)) # Cat 12
iset1 = cbind(iset1,matrix(c(1,1,1,1,1,1,0,0),8,1)) # Cat 13
iset1 = cbind(iset1,matrix(c(1,0,0,0,0,0,0,0),8,1)) # Cat 14
iset1 = cbind(iset1,matrix(c(0,0,1,0,0,0,0,0),8,1)) # Cat 15


input[[1]] = iset1

# Desired outputs
output[[1]] = input[[1]]

# Clear unecessary variables
rm(iset1)

# Create distorted test patterns
# Set parameters
prob<-c(0) # List distortion probabilities
testpp=c()

for (s in 1:length(input)) { # set counter

	# Initiate a zero test pattern matrix with activation placeholders for
	# each unit, training pattern, distortion probability, and run
	testp<-c(rep(0,(dim(input[[s]])[1]*dim(input[[s]])[2]*length(prob)*runs)))
	dim(testp)<-c(dim(input[[s]])[1],dim(input[[s]])[2],length(prob),runs)

	# Fill in testp
	for (p in 1:length(prob)) {                                       # Distortion probability counter
		pdist<-c(rep(1,(100*prob[p])),rep(0,(100-(100*prob[p])))) # Create probability distribution
		for (pat in 1:dim(input[[s]])[2]) {                       # Training pattern counter
			for (patv in 1:dim(input[[s]])[1]) {              # Unit counter
				for (nrun in 1:runs) {                    # Run counter
					pn<-sample(pdist,1)               # Randomly select from distribution

					# Update test activation state
					if (pn==1) {
						if (input[[s]][patv,pat]==1) {
							testp[patv,pat,p,nrun]<-0
						}
						else {
							testp[patv,pat,p,nrun]<-1
						}
					}
					else {
						testp[patv,pat,p,nrun]<-input[[s]][patv,pat]
					} # End update state
				}     # End run counter
			}         # End unit counter

		#Print test pattern to parameter report file
		cat(sprintf("PART 2B SET %.0f: Pattern %.0f with distortion probability: %.1f\n",set,pat,prob[p]),file=parrepfilepath,append=TRUE)
		write.table(testp[,pat,p,],sep="\t",file=parrepfilepath,append=TRUE,col.names=FALSE)
		cat("\n",file=parrepfilepath,append=TRUE)

		} 	# End training pattern counter
	} 		# End distortion probability counter
	testpp[[s]]=testp
} # End set counter


# Initiate list of report variables for each network
bnepoch = c() # No. of epochs to settle
bferr   = c() # Error between final and desired output per network
sferr   = c() # Error between final and desired output per set
srpfa   = c() # Final test activation state per set
bsrpfa  = c() # Final test activation state per network


cat("PART 2B: Final test activation states\n",file=parrepfilepath,append=TRUE)

# Perform training and tests
for (b in 1:length(bnet)) {	# Network counter
	
	# Initiate matrix no. of epochs to settle per set per run
	nepochs      = c(rep(0,length(input)*runs))
	dim(nepochs) = c(length(input),runs)
	
	for (s in 1:length(input)) {	# Set counter
	
		testp = testpp[[s]]

		# Initiate report variables: Error between final output and desired output
		ferr      = c(rep(0,dim(testp)[2]*dim(testp)[3]*dim(testp)[4]))
		dim(ferr) = c(dim(testp)[2],dim(testp)[3],dim(testp)[4])
		rpfa      = c()
		
		# Train and test network
		for (r in 1:dim(testp)[4]) {
		
			# Train
			trainnet_perceptron(Wtr[[r]],train[[s]],output[[s]],bnet[[b]],epochs,err,eta,mu)
			
			# Store variables
			nepochs[s,r]=tepoch
			Wtr[[r]]=Wt[[length(Wt)]]
			
			pfa=c()

			# Test
			for (p in 1:dim(testp)[3]) {
				fa=c()
				for (i in 1:dim(testp)[2]) {
					testnet_perceptron(Wt[[length(Wt)]],as.matrix(testp[,i,p,r]),bnet[[b]])
					
					# Store final rounded test activation state
					fa=cbind(fa,ar[[length(ar)]])
					cat(sprintf("Net %.0f, Set %.0f, Distortion Prob %.1f, Pattern %.0f, Run %.0f\n",b,s,prob[p],i,r),file=parrepfilepath,append=TRUE)
					for (la in 1:length(ar)) {
						cat(ar[[la]],file=parrepfilepath,append=TRUE)
						cat("\n",file=parrepfilepath,append=TRUE)
					}
					cat("\n",file=parrepfilepath,append=TRUE)
					
					# Calculate ferr from actual final test activation states
					d             = as.matrix(output[[s]][,i])-as.matrix(a[[length(a)]])
					ferr[i,p,r]   = sum(d^2)
					
				}
				pfa[[p]]=fa
			}
			rpfa[[r]]=pfa
		}
		
		srpfa[[s]] = rpfa
		sferr[[s]] = ferr

	} # End set counter
	
	# Update report variables
	bnepoch[[b]] = nepochs
	bferr[[b]]   = sferr
	bsrpfa[[b]]  = srpfa
	
} # End network counter



#Print reports
cat("PART 2B: GENERALIZATION\n",file=outputfilepath,append=TRUE)
for (set in 1:length(input)) {
	cat(sprintf("Set %.0f: Mean (SD) number of epochs required to reach error threshold for each network type\n",set),file=outputfilepath,append=TRUE)
	for (net in 1:length(bnepoch)) {
		cat(mean(bnepoch[[net]][set,]),file=outputfilepath,append=TRUE)
		cat("\t(",file=outputfilepath,append=TRUE)
		cat(sd(bnepoch[[net]][set,]),file=outputfilepath,append=TRUE)
		cat(")\n",file=outputfilepath,append=TRUE)
	}
	cat("\n",file=outputfilepath,append=TRUE)
}

for (set in 1:length(input)) {
	cat(sprintf("Set %.0f: Mean (SD) error between desired and actual output for each network type\n",set),file=outputfilepath,append=TRUE)
	for (net in 1:length(bnepoch)) {
		cat(sprintf("Network type %.0f\n",net),file=outputfilepath,append=TRUE)
		for (pn in 1:length(prob)) {
			cat(sprintf("For distortion probability %.1f (rows: training pattern)\n",prob[pn]),file=outputfilepath,append=TRUE)
			for (pat in 1:dim(as.matrix(input[[set]]))[2]) {
				cat(mean(bferr[[net]][[set]][pat,pn,]),file=outputfilepath,append=TRUE)
				cat("\t(",file=outputfilepath,append=TRUE)
				cat(sd(bferr[[net]][[set]][pat,pn,]),file=outputfilepath,append=TRUE)
				cat(")\n",file=outputfilepath,append=TRUE)
			}
			cat("\n",file=outputfilepath,append=TRUE)
		}
	}
}




# PART 3: RULES
#-----------------------------------------------------------------

#Print network types to parameter report file
cat("PART 3: Network type units and layers, input(left)-hidden(middle)-output(right)\n",file=parrepfilepath,append=TRUE)
for (net in 1:length(bnet)) {
	cat(sprintf("Network type %.0f:\t",net),file=parrepfilepath,append=TRUE)
	cat(bnet[[net]],file=parrepfilepath,append=TRUE)
	cat("\n",file=parrepfilepath,append=TRUE)
}
cat("\n",file=parrepfilepath,append=TRUE)
cat("PART 3: Test Patterns, rows(units), columns(runs)\n",file=parrepfilepath,append=TRUE)


# Create training input and outputs
# Set parameters
input  = c()
	
# Set1
# Inputs
iset1 =             matrix(c(0,0,0,0,0,0,0,1),8,1)  # Cat 14

input[[1]]  = iset1
output[[1]] = iset1

# Clear unecessary variables
rm(iset1)

# Create distorted test patterns
# Set parameters
prob<-c(0) # List distortion probabilities
testpp=c()

for (s in 1:length(input)) {

	# Initiate a zero test pattern matrix with activation placeholders for
	# each unit, training pattern, distortion probability, and run
	testp<-c(rep(0,(dim(input[[s]])[1]*dim(input[[s]])[2]*length(prob)*runs)))
	dim(testp)<-c(dim(input[[s]])[1],dim(input[[s]])[2],length(prob),runs)

	# Fill in testp
	for (p in 1:length(prob)) {                                       # Distortion probability counter
		pdist<-c(rep(1,(100*prob[p])),rep(0,(100-(100*prob[p])))) # Create probability distribution
		for (pat in 1:dim(input[[s]])[2]) {                       # Training pattern counter
			for (patv in 1:dim(input[[s]])[1]) {              # Unit counter
				for (nrun in 1:runs) {                    # Run counter
					pn<-sample(pdist,1)               # Randomly select from distribution

					# Update test activation state
					if (pn==1) {
						if (input[[s]][patv,pat]==1) {
							testp[patv,pat,p,nrun]<-0
						}
						else {
							testp[patv,pat,p,nrun]<-1
						}
					}
					else {
						testp[patv,pat,p,nrun]<-input[[s]][patv,pat]
					} # End update state
				}     # End run counter
			}         # End unit counter

		#Print test pattern to parameter report file
		cat(sprintf("PART 3 SET %.0f: Pattern %.0f with distortion probability: %.1f\n",s,pat,prob[p]),file=parrepfilepath,append=TRUE)
		write.table(testp[,pat,p,],sep="\t",file=parrepfilepath,append=TRUE,col.names=FALSE)
		cat("\n",file=parrepfilepath,append=TRUE)

		} 	# End training pattern counter
	} 		# End distortion probability counter
	testpp[[s]]=testp
} # End set counter

# Initiate list of report variables for each network
bferr   = c() # Error between final and desired output per network
sferr   = c() # Error between final and desired output per set
srpfa   = c() # Final test activation state per set
bsrpfa  = c() # Final test activation state per network

cat("PART 3: Final test activation states\n",file=parrepfilepath,append=TRUE)

# Perform training and tests
for (b in 1:length(bnet)) {	# Network counter	
	for (s in 1:length(input)) {	# Set counter
	
		testp = testpp[[s]]

		# Initiate report variables: Error between final output and desired output
		ferr      = c(rep(0,dim(testp)[2]*dim(testp)[3]*dim(testp)[4]))
		dim(ferr) = c(dim(testp)[2],dim(testp)[3],dim(testp)[4])
		rpfa      = c()
		
		# Test network
		for (r in 1:dim(testp)[4]) {
			
			pfa=c()

			# Test
			for (p in 1:dim(testp)[3]) {
				fa=c()
				for (i in 1:dim(testp)[2]) {
					testnet_perceptron(Wtr[[r]],as.matrix(testp[,i,p,r]),bnet[[b]])
					
					# Store final rounded test activation state
					fa=cbind(fa,ar[[length(ar)]])
					cat(sprintf("Net %.0f, Set %.0f, Distortion Prob %.1f, Pattern %.0f, Run %.0f\n",b,s,prob[p],i,r),file=parrepfilepath,append=TRUE)
					for (la in 1:length(ar)) {
						cat(ar[[la]],file=parrepfilepath,append=TRUE)
						cat("\n",file=parrepfilepath,append=TRUE)
					}
					cat("\n",file=parrepfilepath,append=TRUE)
					
					# Calculate ferr from actual final test activation states
					d             = as.matrix(output[[s]][,i])-as.matrix(a[[length(a)]])
					ferr[i,p,r]   = sum(d^2)
					
				}
				pfa[[p]]=fa
			}
			rpfa[[r]]=pfa
		}
		
		srpfa[[s]] = rpfa
		sferr[[s]] = ferr

	} # End set counter
	
	# Update report variables
	bferr[[b]]   = sferr
	bsrpfa[[b]]  = srpfa
	
} # End network counter



#Print reports
cat("PART 3: RULES\n",file=outputfilepath,append=TRUE)

for (set in 1:length(input)) {
	cat(sprintf("Set %.0f: Mean (SD) error between desired and actual output for each network type\n",set),file=outputfilepath,append=TRUE)
	for (net in 1:length(bnepoch)) {
		cat(sprintf("Network type %.0f\n",net),file=outputfilepath,append=TRUE)
		for (pn in 1:length(prob)) {
			cat(sprintf("For distortion probability %.1f (rows: training pattern)\n",prob[pn]),file=outputfilepath,append=TRUE)
			for (pat in 1:dim(as.matrix(input[[set]]))[2]) {
				cat(mean(bferr[[net]][[set]][pat,pn,]),file=outputfilepath,append=TRUE)
				cat("\t(",file=outputfilepath,append=TRUE)
				cat(sd(bferr[[net]][[set]][pat,pn,]),file=outputfilepath,append=TRUE)
				cat(")\n",file=outputfilepath,append=TRUE)
			}
			cat("\n",file=outputfilepath,append=TRUE)
		}
	}
}





# Close files
close(outputfile)
close(parrepfile)

# Clear workspace
rm(list=ls())
