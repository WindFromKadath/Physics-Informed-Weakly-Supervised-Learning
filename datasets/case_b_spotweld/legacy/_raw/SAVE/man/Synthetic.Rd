\name{Synthetic}
\alias{synthfield}
\alias{synthmodel}
\docType{data}
\title{
Synthetic data
}
\description{
Synthetic data from a real experiment (\code{synthfield}) and another one containing observations from the simulator (i.e. the computer model, \code{synthmodel}).
}
\usage{
	data(synthfield)
	
	data(synthmodel)
}
\format{
  Two data frame with observations on the following variables.
  \describe{
    \item{\code{y}}{Response}
    \item{\code{x}}{Controllable input}
    \item{\code{v}}{Calibration variable (only in \code{synthmodel})}
  }
}

\keyword{datasets}
