\name{normal,uniform}
\alias{normal}
\alias{uniform}
\title{ Auxiliary functions. }
\description{ Auxiliary functions. }
\usage{
uniform(var.name, lower, upper)

normal(var.name, mean, sd, lower, upper)
}
\arguments{
  \item{var.name}{The name of the variable whose prior distribution is being specified.}
  \item{lower}{The lower prior bound for \code{var.name}.}
  \item{upper}{The upper prior bound for \code{var.name}.}
  \item{mean}{The prior mean for \code{var.name}.}
  \item{sd}{The prior standard deviation for \code{var.name}.}
}  
     
\details{ These functions are intended as a user-friendly way of specifying the type of priors assumed for the calibration parameters in the function \code{bayesfit}.

For example, in a problem with calibration parameters named "delta1" and "shift", a uniform(0,1) prior for "delta1" and for "shift" a normal density with mean 2 and standard deviation 1 truncated to the interval (0,3), the prior should be specified as\cr

prior=c(uniform(var.name="delta1", lower=0, upper=1),\cr
	normal(var.name="shift", mean=2, sd=1, lower=0, upper=3))
	
 }

