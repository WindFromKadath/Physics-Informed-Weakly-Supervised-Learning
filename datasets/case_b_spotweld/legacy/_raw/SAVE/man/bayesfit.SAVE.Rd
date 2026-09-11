\name{bayesfit}
\alias{bayesfit}
\alias{bayesfit.SAVE}
\alias{bayesfit,SAVE-method}
\alias{bayesfit}

\title{ Bayesian fit }
\description{ Bayesian analysis of a computer model: obtaining the posterior distribution of the unknown parameters in the statistical model of Bayarri et al. (2007). }
\usage{
\S4method{bayesfit}{SAVE}(object, prior, mcmcMultmle=1, prob.prop=0.5,
         method=2,n.iter, nMH=20, n.burnin=0, n.thin=1, verbose=FALSE, ...)}
\arguments{
  \item{object}{An object of class \code{SAVE} normally produced by a call to function \code{\link{SAVE}}.}
  \item{prior}{The prior distribution assumed for the calibration parameters. This should be specified concatenating, \code{c}, calls to functions \code{\link{uniform}} and/or \code{\link{normal}}. See details below.}
  \item{mcmcMultmle}{A factor that is used to specify the prior for lambdaF and lambdaM (the precision parameters in the field and the bias, respectively). See details below.}
  \item{prob.prop}{The probability of proposing from the prior in the Metropolis-Hastings algorithm. See details below.}
  \item{method}{Method implemented in the Gibbs sampling. See details below.}
  \item{n.iter}{Number of total simulations. See details below.}
  \item{nMH}{Number of Metropolis-Hastings steps. See details below.} 
  \item{n.burnin}{The number of iterations at the beginning of the MCMC that are thrown away. See details below.} 
  \item{n.thin}{The thinning to be applied to the resulting MCMC sample. See details below.}
  \item{verbose}{A \code{logical} value indicating the level of output as the function runs.}
  \item{...}{Extra arguments to be passed to the function (still not implemented).}
}  
     
\details{The parameters in the statistical model which are treated as unknown are lambdaB (the precision of the bias, also called discrepancy term); lambdaF (the precision of the error) and the vector of calibration inputs. The function \code{bayesfit} provides a sample from the posterior distribution of these parameters using a particular MCMC strategy. This function depends on two types of arguments detailed below: those defining the prior used, and those providing details for the MCMC sampling.
	
About the prior: the prior for lambdaB and lambdaF is the product of two independent exponential densities with the means being equal to the corresponding maximum likelihood estimates times the factor specified in \code{mcmcMultmle}. Notice that the prior variance increases quadratically with this factor, and the prior becomes less informative as the factor increases.

The prior for each calibration parameter can be either a uniform distribution or a truncated normal and should be specified concatenating calls to \code{\link{uniform}} and/or \code{\link{normal}}. For example, in a problem with calibration parameters named "delta1" and "shift", a uniform(0,1) prior for "delta1" and for "shift" a normal density with mean 2 and standard deviation 1 truncated to the interval (0,3), the prior should be specified as\cr

prior=c(uniform(var.name="delta1", lower=0, upper=1),\cr
	normal(var.name="shift", mean=2, sd=1, lower=0, upper=3))

About the MCMC. The algorithm implemented is based on a Gibbs sampling scheme. If \code{method=2} then the emulator of the computer model and the bias are integrated out (analytically) and only the full conditionals for lambdaB, lambdaF and calibration parameters are sampled from. Else, if \code{method=1} the computer model and the bias are part of the sampling scheme. The calibration parameters are sampled from their full conditional distribution using a Metropolis-Hastings algorithm with candidate samples proposed from a mixture of the prior specified and a uniform centered on the last sampled value. Here, the probability of a proposal coming from the prior is set by \code{prob.prop}. The Metropolis-Hastings algorithm is run \code{nMH} times before each sample is accepted. The default and preferred method is \code{method=2}.

The MCMC is run a total of \code{n.iter} iterations, of which the first \code{n.burnin} are discarded. The remaining samples are thinned using the number specified in \code{n.thin}.  
 }

\value{
  \code{SAVE} returns a copy of the \code{SAVE} object used as argument to the function, but with the following slots filled (or replaced if they where not empty)
 \describe{ 
    \item{\code{method}:}{the value given to \code{method}.}
    \item{\code{mcmcMultmle}:}{the value given to \code{mcmcMultmle}.}
    \item{\code{n.iter}:}{the value given to \code{n.iter}.}
    \item{\code{nMH}:}{The value given to \code{nMH}.}
    \item{\code{mcmcsample}:}{A named \code{matrix} with the simulated samples from the posterior distribution.}
    \item{\code{bayesfitcall}:}{The \code{call} to \code{bayesfit}.}
  }
  
  }

\references{
Palomo J, Paulo R, Garcia-Donato G (2015). SAVE: An R Package for the Statistical Analysis of Computer Models. Journal of Statistical Software, 64(13), 1-23. Available from http://www.jstatsoft.org/v64/i13/

Bayarri MJ, Berger JO, Paulo R, Sacks J, Cafeo JA, Cavendish J, Lin CH, Tu J (2007). A Framework for Validation of Computer Models. Technometrics, 49, 138-154.

Craig P, Goldstein M, Seheult A, Smith J (1996). Bayes linear strategies for history matching of hydrocarbon reservoirs. In JM Bernardo, JO Berger, AP Dawid, D Heckerman, AFM Smith (eds.), Bayesian Statistics 5. Oxford University Press: London. (with discussion).

Higdon D, Kennedy MC, Cavendish J, Cafeo J, Ryne RD (2004). Combining field data and computer simulations for calibration and prediction. SIAM Journal on Scientific Computing, 26, 448-466.

Kennedy MC, O Hagan A (2001). Bayesian calibration of computer models (with discussion). Journal of the Royal Statistical Society B, 63, 425-464.

Roustant O., Ginsbourger D. and Deville Y. (2012).
  DiceKriging, DiceOptim: Two R Packages for the Analysis of Computer
  Experiments by Kriging-Based Metamodeling and Optimization. Journal
  of Statistical Software, 51(1), 1-55.
  
}


\author{Jesus Palomo, Rui Paulo and Gonzalo Garcia-Donato}
  
\seealso{ \code{\link[=plot,SAVE-method]{plot}},
	  	  \code{\link[=predictreality.SAVE]{predictreality}},
	      \code{\link[=validate,SAVE-method]{validate}}
}
	
\examples{
\dontrun{
library(SAVE)

#############
# load data
#############

data(spotweldfield,package='SAVE')
data(spotweldmodel,package='SAVE')

##############
# create the SAVE object which describes the problem and
# compute the corresponding mle estimates
##############

gfsw <- SAVE(response.name="diameter", controllable.names=c("current", "load", "thickness"), 
			 calibration.names="tuning", field.data=spotweldfield, 
			 model.data=spotweldmodel, mean.formula=~1, 
			 bestguess=list(tuning=4.0))

##############
# obtain the posterior distribution of the unknown parameters 
##############

gfsw <- bayesfit(object=gfsw, prior=c(uniform("tuning", upper=8, lower=0.8)),
				 n.iter=20000, n.burnin=100, n.thin=2)

# summary of the results
summary(gfsw)
	
	}
}

