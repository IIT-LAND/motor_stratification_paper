ns_gene_decode <- function (image, tissue, fname2save, fdr_thresh = 0.05, pval_thresh = NULL){
  # ns_gene_decode - function to do Neurosynth Gene Expression Decoding
  #
  # INPUT
  #   image = number of image to use on Neurovault
  #   tissue = either "full", "cortex", or "subcortex"
  #   fname2save = full file path and name to csv file you want to save results to 
  #

  # Load libraries
  library(rjson)


  # load the json from Neurosynth
  url_name = file.path("https://neurovault.org/images",image,"gene_expression",sprintf("json?mask=%s",tissue))
  data = fromJSON(file = url_name)

  # I don't like lists so convert to a usable dataframe (there's probably a better way to do this...)
  df = data.frame(matrix(t(unlist(data$data)), nrow=length(data$data), byrow=T))
  colnames(df) = c("symbol","page?","name","t","p","p_corr","var_explained","var_sd")
  
  # now make sure columns have the correct format
  df$t <- as.numeric(as.character(df$t))
  df$p <- as.numeric(as.character(df$p))
  df$p_corr <- as.numeric(as.character(df$p_corr))
  df$var_explained <- as.numeric(as.character(df$var_explained))
  df$var_sd <- as.numeric(as.character(df$var_sd))
  
  # grab subset of genes with positive t-stat and p_corr<=fdr_thresh
  pos_df_fdr = subset(df,df$t>0 & df$p_corr<=fdr_thresh)
  # grab subset of genes with negative t-stat and p_corr<=fdr_thresh
  neg_df_fdr = subset(df,df$t<0 & df$p_corr<=fdr_thresh)

  # write out results
  write.csv(df, file = fname2save)
  
  # write out only if there is stuff surviving the FDR correction
  if (dim(pos_df_fdr)[1]!=0){
    write.csv(pos_df_fdr, 
              file = file.path(dirname(fname2save), 
                               sprintf("%s_%s_%s_fdr%.02f.csv",image,tissue,"pos_t",fdr_thresh)))
  }
  
  # write out only if there is stuff surviving the FDR correction
  if (dim(neg_df_fdr)[1]!=0){
    write.csv(neg_df_fdr, 
              file = file.path(dirname(fname2save), 
                               sprintf("%s_%s_%s_fdr%.02f.csv",image,tissue,"neg_t",fdr_thresh)))
  }

  
  
  if (!is.null(pval_thresh)){
    # grab subset of genes with positive t-stat and p<=pval_thresh
    pos_df_pval = subset(df,df$t>0 & df$p<=pval_thresh)
    # grab subset of genes with negative t-stat and p<=pval_thresh
    neg_df_pval = subset(df,df$t<0 & df$p<=pval_thresh)

    if (dim(pos_df_pval)[1]!=0){
      write.csv(pos_df_pval, 
                file = file.path(dirname(fname2save), 
                                 sprintf("%s_%s_%s_pval%.04f.csv",image,tissue,"pos_t",pval_thresh)))
    }
    
    if (dim(neg_df_pval)[1]!=0){
      write.csv(neg_df_pval, 
                file = file.path(dirname(fname2save), 
                                 sprintf("%s_%s_%s_pval%.04f.csv",image,tissue,"neg_t",pval_thresh)))
    }
    
  }
  
} # ns_gene_decode.R