\name{predictcode}
\alias{predictcode}
%\alias{predictcode.SAVE}
\alias{summary.predictcode.SAVE}
\alias{summary,predictcode.SAVE-method}
\alias{show.summary.predictcode.SAVE}
\alias{plot,predictcode.SAVE-method}
\alias{predictcode.SAVE-class}
\alias{predictcode,SAVE-method}
\alias{show,predictcode.SAVE-method}
\alias{show,summary.predictcode.SAVE-method}


%\alias{summary.predictcode.SAVE}
%\alias{show.summary.predictcode.SAVE}
\alias{plot.predictcode.SAVE}
%\alias{predictcode.SAVE-class}

%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Predict values of the computer model at new input points }
\description{
The emulator of the computer model fitted by \code{SAVE} is used to predict values of the model at new input points.
}
\usage{
\S4method{predictcode}{SAVE}(object, newdesign, n.iter=1000, sampledraws=T, tol=1e-10, verbose=FALSE)

\S4method{summary}{predictcode.SAVE}(object)

\S4method{show}{summary.predictcode.SAVE}(object)

\S4method{plot}{predictcode.SAVE}(x, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{ An object of the corresponding signature. }
  \item{newdesign}{ A named matrix containing the points (calibration and controllable inputs) where predictions are to be performed. Column names should contain both the \code{object@controllablenames} and \code{object@calibrationnames}}  
  \item{n.iter}{The number of simulations that are to be drawn from the emulator (see details below)}
  \item{sampledraws}{If TRUE a sample of size \code{n.iter} is obtained from the emulator. If FALSE only the covariance matrix and the mean of the emulator are returned.}
  \item{tol}{The tolerance in the Cholesky decomposition}
  \item{verbose}{A \code{logical} value indicating the level of output as the function runs.}
  \item{...}{Extra arguments to be passed to the function (still not implemented).}
  \item{x}{ An object of class \code{predictcode.SAVE} }

}

\value{
	Returns an S4 object of the class \code{predictcode.SAVE} that contains the following slots:
  \item{newdesign }{ A copy of the design. }
  \item{samples }{ The matrix that contains the simulations (see details). }
  \item{mle }{ A copy of the maximum likelihood estimate \code{object@mle}. }
  \item{predictcodecall }{ The call to this function. }
  \item{modelmean }{ The mean of the emulator (see details) at the new design \code{newdesign}. }
  \item{covmat }{ The covariance matrix of the emulator (see details) at the new design \code{newdesign}. }
  
}

\details{
The emulator of the computer model fitted by \code{SAVE} evaluated at the new input points specified in \code{newdesign} is a multivariate normal. Then \code{predictcode} computes the mean, the covariance matrix and, if \code{sampledraws=TRUE}, a simulated sample of size \code{n.iter} from this multivariate normal. A pivotal Cholesky decomposition algorithm is used in the simulation of the samples and \code{tol} is a tolerance parameter in this algorithm.

The object created can be explored with the functions \code{plot} and \code{summary}. The first function plots a graphic with the mean and 95\% tolerance bounds of the emulator at each of the new input points. Furthermore, \code{summary} prints a matrix with the mean of the emulator at each new input point, the associated standard deviation, and 95\% tolerance bounds.
}

\author{ Jesus Palomo, Rui Paulo and Gonzalo Garcia-Donato. }

\seealso{  \code{\link{SAVE}} }

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

##########
# emulate the output of the model using predictcode
##########

# construct design at which to emulate the model
u <- 3.2
load <- c(4.0,5.3)
curr <- seq(from=20,to=30,length=20)
g <- c(1,2)

xnewpure <- expand.grid(curr,load,g)
xnewpure <- cbind(xnewpure,rep(u,dim(xnewpure)[1]))
names(xnewpure) <- c("current","load","thickness","tuning")
xnewpure <- as.data.frame(xnewpure)

pcsw<- predictcode(object=gfsw, newdesign=xnewpure, n.iter=20000, tol=1.E-12)
	
#A summary of the emulation:
summary(pcsw)	

#A plot of the emulation
plot(pcsw)
}	
	
}
