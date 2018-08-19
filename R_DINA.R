setwd("D:/DINA/math2015/FrcSub")
data <- read.csv("data.csv")
q <- read.csv("q.csv")
datai <- nrow(data)
dataj <- ncol(data)
qi <- nrow(q)
qj <- ncol(q)
sg = matrix(0.01,qi,2)
continueSG = TRUE
nTrueOrFalse <- function(q,l) {
    if (FALSE %in% (as.numeric(q[1,]) == as.numeric(c(0,0,0,1,0,1,1,0)))){
        return(0)}
    else{
        return (1)}
}
while(continueSG == TRUE){
      IL = matrix(1,datai, 2 ** qj)
      for( l in 1:2 ** qj){
          for( i in 1:datai){
              for (j in 1:dataj){
                  nl = nTrueOrFalse(q[j,], l)
                  if (nl == 1){
                      if( data[i,j] == 1){
                          IL[i,l] = IL[i,l] *(1 - sg[j,1])}
                      else{
                          IL[i,l] = IL[i,l] *sg[j,1]}
			}
                  else{
                      if( data[i,j] == 1){
                          IL[i,l] = IL[i,l] *sg[j,2]}
                      else{
                          IL[i,l] = IL[i,l] *(1 - sg[j,2])}
			}
		  }
		}
      	cat('l总数:',2 ** qj,',i总数:',datai,',l:',l)
	}
      print("IL是训练集学生，所有技能模式的似然概率矩阵")
      print(IL)
	break
}
