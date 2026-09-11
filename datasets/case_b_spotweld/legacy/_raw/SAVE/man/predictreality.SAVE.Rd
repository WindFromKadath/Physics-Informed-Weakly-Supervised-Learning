\name{predictreality}
\alias{predictreality}
\alias{predictreality.SAVE}
\alias{predictreality.SAVE-class}
\alias{predictreality,SAVE-method}
\alias{summary.predictreality.SAVE}
\alias{summary,predictreality.SAVE-method}
\alias{show.summary.predictreality.SAVE}
\alias{show,predictreality.SAVE-method}
\alias{show,summary.predictreality.SAVE-method}



%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Predict values of reality at new input points }
\description{
The emulator of the computer model and the Bayesian fit are used to produce samples from the posterior predictive distribution of the computer model and bias function evaluated at the new input points. Then, bias-corrected predictions of the response (reality) are produced by adding these two samples (model+bias).
}
\usage{
\S4method{predictreality}{SAVE}(object, newdesign, n.burnin=0, n.thin=1, tol=1E-10, verbose=FALSE, ...)

\S4method{summary}{predictreality.SAVE}(object)

\S4method{show}{summary.predictreality.SAVE}(object)

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{ An object of the corresponding signature. }
  \item{newdesign}{ A named matrix containing the points (controllable
inputs) where predictions are to be performed. Column names should contain
the \code{object@controllablenames}. This parameter should be set to NULL
in the situation with constant controllable inputs.}
  \item{n.burnin}{ The burnin to be applied (see details below).} 
  \item{n.thin}{ The thinin to be applied (see details below).}   
  \item{tol}{ The tolerance in the Cholesky decomposition.}
  \item{verbose}{A \code{logical} value indicating the level of output as the function runs.}
  \item{...}{ Extra arguments to be passed to the function (still not implemented).}
}

\value{
	Returns an S4 object of class \code{predictreality.SAVE} with the following slots:
  \item{modelpred}{A list with the simulations from the posterior distribution of the computer model output evaluated at the new design}
  \item{biaspred }{ A matrix with the simulations from the posterior distribution of the bias function evaluated at the new design. }
  \item{newdesign }{ A copy of the design given as argument. }
  \item{predictrealitycall }{ The call to the function. }

}

\details{
Draws from the posterior predictive distribution of the computer model and bias at a given set of controllable inputs are simulated using the MCMC sample from the posterior distribution of the parameters of the model stored in \code{object@mcmcsample}. This sample can be thinned by 
\code{n.thin} and/or the first \code{n.burnin} draws can be discarded.

A preliminary analysis of the resulting sample can be performed with \code{summary} which provides point estimates and tolerance bounds of the predictions.
}

\author{ Jesus Palomo, Rui Paulo and Gonzalo Garcia-Donato. }

\seealso{ \code{\link[=validate,SAVE-method]{validate}} }

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

# summary of the results

summary(gfsw)

##############
# obtain the posterior distribution of the unknown parameters
##############

gfsw <- bayesfit(object=gfsw, prior=c(uniform("tuning", upper=8, lower=0.8)), 
				 n.iter=20000, n.burnin=100, n.thin=2)

#########
# bias-corrected prediction at a set of inputs
# using predictreality
##########

load <- c(4.0,5.3)
curr <- seq(from=20,to=30,length=20)
g <- c(1,2)

xnew<- expand.grid(current = curr, load = load, thickness=g)

# Obtain samples
prsw <- predictreality(object=gfsw, newdesign=xnew, tol=1.E-12)

#Summarize the results:
summary(prsw)

}	
	
	}
