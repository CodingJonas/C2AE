function save_result(p, y, dataset, param1, param2, param3)
    [precision_f1, recall_f1, F1_f1] = f1_score(p, y);
	fprintf('prec = %f, rec = %f, f1 = %f\n', precision_f1, recall_f1, F1_f1);
	
	fid = fopen([dataset, '_result.csv'], 'a');
    fprintf(fid, '%.1f,%.1f,%.1f,%f,%f\n', param1, param2, param3, micro_f1, macro_f1);
    fclose(fid);
end
