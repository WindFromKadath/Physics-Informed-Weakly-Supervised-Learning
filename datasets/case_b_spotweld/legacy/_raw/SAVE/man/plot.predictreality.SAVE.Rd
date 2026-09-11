\name{plot.predictreality.SAVE}
\alias{plot.predictreality.SAVE}
\alias{plot,predictreality.SAVE-method}
\title{
A function for plotting summaries of an object of class \code{predictreality.SAVE}.
}
\description{
Two different plots to summarize graphically the results in an object of class \code{predictreality.SAVE}.
}
\usage{
\S4method{plot}{predictreality.SAVE}(x, option = "trace", ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{An object of class \code{predictreality.SAVE}
  	
}
  \item{option}{One of "biascorr" or "biasfun" (see details)
}
  \item{...}{Additional graphical parameters to be passed}
}
\details{
If \code{option}="biascorr" this function returns a plot with point predictions and 95\% tolerance bounds of reality at the given set of controllable inputs. If \code{option}="biasfun" the plot represents the estimated bias and 95\% credible bounds.
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

#Plot the results:
#Represent reality and tolerance bounds:
plot(prsw, option="biascorr")
#Represent bias and tolerance bounds:
plot(prsw, option="biasfun")


}
}

