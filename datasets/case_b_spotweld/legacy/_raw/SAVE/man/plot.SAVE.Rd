%\name{plot.SAVE}
\name{plot}
\alias{plot.SAVE}
\alias{plot}
\alias{plot,SAVE-method}
\title{
Plots for an object of class \code{SAVE}
}
\description{
Plots are provided to summarize graphically the Bayesian analysis of a computer model.
}
\usage{
\S4method{plot}{SAVE}(x, option = "trace", ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{An object of class \code{SAVE}
  	
}
  \item{option}{One of "trace", "calibration" or "precision"(see details)
}
  \item{...}{Additional graphical parameters to be passed}
}
\details{
Three different plots are implemented. If \code{option}="trace" this function returns the trace plots of the simulated chains. This plot is useful for assessing the convergence of the sampling method. If \code{option}="calibration" this function plots an histogram of the sample obtained from the posterior distribution of the calibration parameters and a line representing the prior assumed. If \code{option}="precision" the histograms and priors correspond to the precision parameters.
}

\author{ Jesus Palomo, Rui Paulo and Gonzalo Garcia-Donato. }

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

#A trace plot of the chains
plot(gfsw, option="trace")
#The histogram of the posterior density of calibration parameters
plot(gfsw, option="calibration")
#The histogram of the posterior density of precision parameters
plot(gfsw, option="precision")

}
}

