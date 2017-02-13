load final/1459x1615/real_aabd_filtp3_weight3_gs_dp1100_jack0_real_aabd_filtp3_weight3_gs_dp1100_jack0_pureB_matrix_cm_directbpwf.mat

% 217x217 - 10
% 353x353 - 11
% 217x353 - 66

% BB
s = 4;

cross    = squeeze(r(66).simd(:,s,:));
auto_217 = squeeze(r(10).simd(:,s,:));
auto_353 = squeeze(r(11).simd(:,s,:));

R = mean(cross,2) ./ sqrt(mean(auto_217,2).*mean(auto_353,2));
l = r(1).l;

semilogx(l,R,'.');xlim([10,1000]);ylim([.5,1.2]);

