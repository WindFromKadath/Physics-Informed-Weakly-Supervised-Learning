\name{SAVE-class}
\docType{class}
\alias{SAVE-class}

\title{ SAVE class }

\description{ S4 class for Statistical Analysis and Validation Engine.}

\section{Objects from the Class}{
Objects of this class are created and initialized with the function \code{\link{SAVE}} that computes the calculations needed for setting up the analysis. These can be completed with the function \code{\link[=bayesfit.SAVE]{bayesfit}} that performs the Bayesian analysis in the SAVE methodology}

\section{Slots}{
	 \describe{
    \item{\code{responsename}:}{Object of class \code{character}. The response name.}
    \item{\code{controllablenames}:}{Object of class \code{character}. The names of the controllable inputs.}
    \item{\code{calibrationnames}:}{Object of class \code{character}. The names of the calibration inputs.}
    \item{\code{constant.controllables}:}{Object of class \code{logical}. Controls whether or not the analysis has constant controllable inputs.}
    \item{\code{df}:}{Object of class \code{matrix}. The field design once the replicates (if any) have been removed.}
    \item{\code{dm}:}{Object of class \code{matrix}. The model design.}
    \item{\code{ym}:}{Object of class \code{numeric}. Model response associated with dm.}
    \item{\code{yf}:}{Object of class \code{numeric}. The field observations.}
    \item{\code{meanformula}:}{Object of class \code{formula}. The formula that specifies the mean function of the emulator of the computer model.}
    \item{\code{mle}:}{The maximum likelihood estimates. This is a \code{list} with three components
    	\describe{
    		\item{\code{thetaM}:}{A \code{numeric} vector containing the estimate of the parameters specifying the covariance structure of the emulator of the computer model. This covariance function has precision lambdaM and a separable correlation function with k(x,y)=exp(-betaM*h^alphaM) where h=abs(x-y). The vector \code{thetaM} is organized as follows: (lambdaM, betaM, alphaM), where betaM and alphaM are named vectors.
    			}
    		\item{\code{thetaL}:}{The \code{numeric} vector of regression coefficients associated with the mean function of the emulator of the computer model}
    		\item{\code{thetaF}:}{A \code{numeric} vector organized as (lambdaB, betaB, alphaB, lambdaF) containing the estimates of lambdaF, the precision of the field measurement error, and of the parameters specifying the Gaussian process prior of the bias function. The covariance function and the parameters follow the same structure as that described for \code{thetaM}}
    		}}    	
    \item{\code{bestguess}:}{A \code{numeric} vector containing the best guess (provided in the call) for the calibration inputs.}
    \item{\code{xm}:}{The model matrix corresponding to the evaluation of the \code{meanformula} at \code{dm}.}
    \item{\code{xf}:}{The model matrix corresponding to the evaluation of the \code{meanformula} at \code{df}.}
    \item{\code{prior}:}{Description of the prior used (empty if if \code{bayesfit} is not run).}
    \item{\code{method}:}{A \code{numeric} object with possible values 1 and 2. Two different MCMC methods have been implemented. If \code{method}=2 then the computer model and bias are integrated out (analytically) before sampling the calibration parameters. If \code{method}=1 then the calibration parameters is sampled from the full conditional. (Empty if if \code{bayesfit} is not run).}
    \item{\code{mcmcMultmle}:}{A positive \code{numeric} object. Priors for the precisions (lambdaM and lambdaB) are exponential distributions centered at the corresponding mle multiplied by \code{mcmcMultmle}. (Empty if if \code{bayesfit} is not run).}
    \item{\code{mcmcsample}:}{A \code{matrix} with the  result of the MCMC sampling after the burnin and thinin has been applied. (Empty if if \code{bayesfit} is not run).}
    \item{\code{wd}:}{A \code{character} with the name of the working directory.}
    \item{\code{call}:}{The \code{call} to \code{SAVE} function to create the object.}
    \item{\code{bayesfitcall}:}{The \code{call} to \code{bayesfit}. (Empty if if \code{bayesfit} is not run).}

    }
  }


\section{Methods}{
  \describe{
    \item{summary}{A summary of the object created. }
    \item{show}{Prints the summary of the object. }
    \item{plot}{See \code{\link[=plot,SAVE-method]{plot}}. }
    \item{predictcode}{See \code{\link[=predictcode]{predictcode}}.}
	\item{bayesfit}{See \code{\link[=bayesfit.SAVE]{bayesfit}}.}
	\item{predictreality}{See \code{\link[=predictreality.SAVE]{predictreality}}.}
	\item{validate}{See \code{\link[=validate,SAVE-method]{validate}}.}
	
	
	 }
}


\author{ J. Palomo, R. Paulo and G. Garcia-Donato }

\seealso{ \code{\link{SAVE}} for more details about how to create a \code{SAVE} object. }

\keyword{classes}
