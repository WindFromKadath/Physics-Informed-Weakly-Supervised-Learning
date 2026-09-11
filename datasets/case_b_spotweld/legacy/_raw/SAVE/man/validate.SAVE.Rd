\name{validate}
%\alias{validate.SAVE}
\alias{validate.SAVE-class}
\alias{validate,SAVE-method}
\alias{validate}
\alias{summary.validate.SAVE}
\alias{summary,validate.SAVE-method}
\alias{show.summary.validate.SAVE}
\alias{plot.validate.SAVE}
\alias{plot,validate.SAVE-method}
\alias{show,summary.validate.SAVE-method}
\alias{show,validate.SAVE-method}


\title{ Validation of a computer model }
\description{Assessing the validity of a computer model at a given set of controllable inputs}
\usage{
\S4method{validate}{SAVE}(object, newdesign, calibration.value="mean", prob=0.90, 
						  n.burnin=0, n.thin=1, tol=1E-10, ...)

\S4method{summary}{validate.SAVE}(object)

\S4method{show}{summary.validate.SAVE}(object)

\S4method{plot}{validate.SAVE}(x, ...)

}
\arguments{
  \item{object}{ An object of corresponding signature. }
  \item{x}{ An object of class \code{validate.SAVE}. }
  \item{newdesign}{ A named matrix containing the points (controllable
inputs) where predictions are to be performed. Column names should contain
the \code{object@controllablenames}. This parameter should be set to NULL
in the situation with constant controllable inputs.}
\item{calibration.value}{ Either the summary of the posterior distribution
of the calibration parameters which is taken as the true value for the
computer model (possible values are "mean" and "median") or a named data
frame with numerical values to be used for the calibration parameters. }
  \item{prob}{ The probability used for the representation of the credible intervals. }
  \item{n.burnin}{ The burn-in to be applied (see details below).} 
  \item{n.thin}{ The thinning to be applied (see details below).}   
  \item{tol}{ The tolerance in the Cholesky decomposition.}
  \item{...}{Additional arguments to be passed, still not implemented. }
}  
     
\value{
Returns an S4 object of class \code{validate.SAVE} with the following slots:
 \describe{ 
 	\item{\code{bayesfitcall}:}{ The \code{call} to \code{bayesfit}. }
	\item{\code{call}:}{ The original \code{call} to \code{SAVE} function. }
	\item{\code{newdesign}:}{ A copy of the design given. }
	\item{\code{validate}:}{ A matrix with the results of the validation analysis (see details below). }
	\item{\code{validatecall}:}{ The call to the \code{validate} function. }
	}
  
  }

\details{ Following the framework for the analysis of computer models by Bayarri et al (2007), validation of a computer model translates to the question: is the computer model producing results that are accurate enough for its intended use?

Answering this question implies the comparison of the responses of the computer model at the `true' value of the calibration parameters (\code{calibration.value}) with reality. This comparison should be performed at the set of controllable inputs that are of interest for the researcher (which in turn represent the `intended use' for the computer model) and that are passed as the argument \code{newdesign} to the function.  

For this comparison to be performed, \code{validation} returns a matrix (\code{@validate}) containing (for each one of the input values in the rows of this matrix) the prediction of reality (column called "bias.corrected") jointly with the estimate of the model response, both with corresponding accompanying tolerance bounds (columns called "tau.bc" and "tau.pm" respectively). These measures should be interpreted as:

Prob(|estimate-real value|<tau)=\code{prob}

Also, the discrepancy between computer model and reality can be assessed through the estimated bias (column called "bias") with the associated 100\code{prob}\% credible interval (columns called "bias.Lower" and "bias.Upper").

In the calculations, the simulated sample from the posterior distribution contained in \code{object@mcmsample} is used. It is possible to discard some of the samples in this matrix by not considering the first \code{n.burnin} and/or thinning by \code{n.thin}.

The results can be conveniently visualized with the functions \code{summary} and \code{plot}.

 }



\references{
Palomo J, Paulo R, Garcia-Donato G (2015). SAVE: An R Package for the Statistical Analysis of Computer Models. Journal of Statistical Software, 64(13), 1-23. Available from http://www.jstatsoft.org/v64/i13/

Bayarri MJ, Berger JO, Paulo R, Sacks J, Cafeo JA, Cavendish J, Lin CH, Tu J (2007). A Framework for Validation of Computer Models. Technometrics, 49, 138-154.
}

\author{Jesus Palomo, Rui Paulo and Gonzalo Garcia-Donato}
  
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

##############
# validate the computer model at chosen set of controllable
# inputs
###############

load <- c(4.0,5.3)
curr <- seq(from=20,to=30,length=20)
g <- c(1,2)

xnew <- expand.grid(current = curr, load = load, thickness=g)

valsw <- validate(object=gfsw,newdesign=xnew,n.burnin=100)

# summary of results
summary(valsw)
# plot results
plot(valsw)


	}
}

