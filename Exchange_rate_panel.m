%////////////////////////////////////////////////////////////////%
%//////-Panel Data model Multi-Country -Model of Exchange Rate (Assessment 1) -///%
%////////////////////////////////////////////////////////////////%

disp('------------------------------------------------------------------');
disp('    PANEL DATA MODELS (Countries Nominal Exchange Rate Effect)    ');
disp('------------------------------------------------------------------');
 
%-----------------------------------------------------------------------------------------
%https://uk.mathworks.com/help/stats/panel-analysis-with-panel-corrected-standard-errors.html
%https://uk.mathworks.com/matlabcentral/fileexchange/46515-multilevel-mixed-effects-modeling-using-matlab
%------------------------------------------------------------------------------------------


% Loading the information to matlab 
clc
clear 
tic
eur=csvread('Euro_m.csv');
unk=csvread('UK_m.csv'); 
jap=csvread('Japan_m.csv');
can=csvread('Canada_m.csv');
den=csvread('Denmark_m.csv');
nor=csvread('Norway_m.csv');
swe=csvread('Sweden_m.csv');
swi=csvread('Switz_m.csv');
kor=csvread('Korea_m.csv');
uns=csvread('US_m.csv');

%Setting the time sample for each country. 
%--IMPORTANT NOTE: here were are going to study with all the countries from
%the data set. In this regard, due Switzaterland has the lower number of
%observation this estimation will has the sample from 1995m1 to 2007m12
%Here could be interesting to observe the dyamics and the effects over the
%exchange rate before the financial crisis in 2008.

t=(1995:0.0833:2007.91667)' %This comand it will be use for setting the time from 1995m1 to 2007m12
t_2=[1995.083:0.0833:2007.91667]'
eur_1=eur(61:216,:); %Here it will be selected only the information from 1995m1 to 2007m12 for the case of the Euro  
unk_1=unk(277:432,:);
jap_1=jap(277:432,:);
can_1=can(277:432,:);
den_1=den(253:408,:);
nor_1=nor(193:348,:);
swe_1=swe(277:432,:);
%Switz already has the benchmark of the time sample
kor_1=kor(181:336,:);
uns_1=uns(277:432,:);

%Data tranformatio. Applying log to exchange rate and price index
eur_1=[log(eur_1(:,1)) eur_1(:,2) eur_1(:,3) log(eur_1(:,4)) eur_1(:,5)  ]  
unk_1=[log(unk_1(:,1)) unk_1(:,2) unk_1(:,3) log(unk_1(:,4)) unk_1(:,5)  ]
jap_1=[log(jap_1(:,1)) jap_1(:,2) jap_1(:,3) log(jap_1(:,4)) jap_1(:,5)  ]
can_1=[log(can_1(:,1)) can_1(:,2) can_1(:,3) log(can_1(:,4)) can_1(:,5)  ]
den_1=[log(den_1(:,1)) den_1(:,2) den_1(:,3) log(den_1(:,4)) den_1(:,5)  ]
nor_1=[log(nor_1(:,1)) nor_1(:,2) nor_1(:,3) log(nor_1(:,4)) nor_1(:,5)  ]
swe_1=[log(swe_1(:,1)) swe_1(:,2) swe_1(:,3) log(swe_1(:,4)) swe_1(:,5)  ]
swi_1=[log(swi(:,1)) swi(:,2) swi(:,3) log(swi(:,4)) swi(:,5)  ]
kor_1=[log(kor_1(:,1)) kor_1(:,2) kor_1(:,3) log(kor_1(:,4)) kor_1(:,5)  ]
uns_1=[ uns_1(:,1) uns_1(:,2) log(uns_1(:,3)) uns_1(:,4)  ]
uns_2=[ uns_1(:,3)  ]

%----Note: The reason  I repean the matrix with the information from each countrie is due helps me to get a better visualization of the tranformation of the data 

% Storage (creating) matrix of nominal exchange rate
mtx_fx=[ eur_1(:,1) unk_1(:,1) jap_1(:,1) can_1(:,1) den_1(:,1) nor_1(:,1) swe_1(:,1) swi_1(:,1) kor_1(:,1) ]

%Heterocedastick test for fx (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
for i = 1:size(mtx_fx,2)
    het_s(i,2)=archtest(mtx_fx(:,i));
end

%Delta S natural logarithm of the nominal exchange rate for equation 1
%(Transformation using difference for the exchange rate)
for i=1:size(mtx_fx,2)
    for ii=2:size(mtx_fx,1)
        mtx_diff(ii-1,i)=mtx_fx(ii,i)-mtx_fx(ii-1,i);
    end
end    

%-- Estimation of z for equation 1 matrix containing log of the price index
mtx_pr=[ eur_1(:,4) unk_1(:,4) jap_1(:,4) can_1(:,4) den_1(:,4) nor_1(:,4) swe_1(:,4) swi_1(:,4) kor_1(:,4) ]
%Staring in the matrix from 2 (second) row since i lost one observation due
%the tranformation 0 (Transformation of the price index 
for i=1:size(mtx_pr,2)
    mtx_z(:,i)=uns_2(2:end,1)-(mtx_pr(2:end,i)-mtx_fx(2:end,i));
end

%Heterocedasticity test for z (Inflation) (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
for i =1:size(mtx_z,2)
    het_z(i,2)=archtest(mtx_z(:,i));
end

%Size of the vector
i  = size(mtx_diff,1)*size(mtx_diff,2);
ii = size(mtx_z,1)*size(mtx_z,2);
%Using the pooled data from the two matrixes or adjusting the information in a Panel Data Structure (where the order is Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
%--Delta s
s_pool=reshape(mtx_diff,i,1);
%--Pooled Z
z_pool=reshape(mtx_z,ii,1);


%/////-------Graph Inspection

figure;
plot(t, eur_1(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate Euro/US (1995m1-2007m12)')
figure;
plot(t, unk_1(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate UK/US (1995m1-2007m12)')
figure;
plot(t, jap_1(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate JAP/US (1995m1-2007m12)')
figure;
plot(t, can_1(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate CANADA/US (1995m1-2007m12)')
figure;
plot(t, den_1(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate DENMARK/US (1995m1-2007m12)')
figure;
plot(t, nor_1(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate NORWAY/US (1995m1-2007m12)')
figure;
plot(t, swe_1(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate SWEDEN/US (1995m1-2007m12)')
figure;
plot(t, swi(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate SWITZERLAND/US (1995m1-2007m12)')
figure;
plot(t, kor_1(:,1), 'LineWidth',1.9), title('Graph: Nominal Exchange Rate KOREA/US (1995m1-2007m12)')

%--Ploting the variables in a single figure 
% Nominal Exchange Rate: Order (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
figure;
subplot(3,3,1)
plot(t,eur_1(:,1), 'LineWidth', 1.5), title('Euro/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,2)
plot(t,unk_1(:,1), 'LineWidth', 1.5), title('UK/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,3)
plot(t,jap_1(:,1), 'LineWidth', 1.5), title('JAP/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,4)
plot(t,can_1(:,1), 'LineWidth', 1.5), title('CANADA/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,5)
plot(t,den_1(:,1), 'LineWidth', 1.5), title('DENMARK/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,6)
plot(t,nor_1(:,1), 'LineWidth', 1.5), title('NORWAY/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,7)
plot(t,swe_1(:,1), 'LineWidth', 1.5), title('SWEDEN/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,8)
plot(t,swi(:,1), 'LineWidth', 1.5), title('SWITZERLAND/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,9)
plot(t,kor_1(:,1), 'LineWidth', 1.5), title('KOREA/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
sgtitle('Graph: Nominal Exchange Rate (1995m1-2007m12)')

% PI: Order (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
figure;
subplot(3,3,1)
plot(t,eur_1(:,4), 'LineWidth', 1.5), title('Eur', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,2)
plot(t,unk_1(:,4), 'LineWidth', 1.5), title('UK', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,3)
plot(t,jap_1(:,4), 'LineWidth', 1.5), title('JAP', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,4)
plot(t,can_1(:,4), 'LineWidth', 1.5), title('CANADA', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,5)
plot(t,den_1(:,4), 'LineWidth', 1.5), title('DENMARK', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,6)
plot(t,nor_1(:,4), 'LineWidth', 1.5), title('NORWAY', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,7)
plot(t,swe_1(:,4), 'LineWidth', 1.5), title('SWEDEN', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,8)
plot(t,swi(:,4), 'LineWidth', 1.5), title('SWITZERLAND', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,9)
plot(t,kor_1(:,4), 'LineWidth', 1.5), title('KOREA', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
sgtitle('Graph: Price Index (1995m1-2007m12)')

% DIFF Nominal Exchange Rate : Order (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
figure;
subplot(3,3,1)
plot(t_2,mtx_diff(:,1), 'LineWidth', 1.5), title('Eur/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,2)
plot(t_2,mtx_diff(:,2), 'LineWidth', 1.5), title('UK/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,3)
plot(t_2,mtx_diff(:,3), 'LineWidth', 1.5), title('JAP/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,4)
plot(t_2,mtx_diff(:,4), 'LineWidth', 1.5), title('CANADA/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,5)
plot(t_2,mtx_diff(:,5), 'LineWidth', 1.5), title('DENMARK/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,6)
plot(t_2,mtx_diff(:,6), 'LineWidth', 1.5), title('NORWAY/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,7)
plot(t_2,mtx_diff(:,7), 'LineWidth', 1.5), title('SWEDEN/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,8)
plot(t_2,mtx_diff(:,8), 'LineWidth', 1.5), title('SWITZERLAND/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,9)
plot(t_2,mtx_diff(:,9), 'LineWidth', 1.5), title('KOREA/US', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
sgtitle('Graph: Diff of Nominal Exchange Rate (1995m2-2007m12)')



% Real Exchange Rate : Order (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
figure;
subplot(3,3,1)
plot(t_2,mtx_z(:,1), 'LineWidth', 1.5), title('Eur', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,2)
plot(t_2,mtx_z(:,2), 'LineWidth', 1.5), title('UK', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,3)
plot(t_2,mtx_z(:,3), 'LineWidth', 1.5), title('JAP', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,4)
plot(t_2,mtx_z(:,4), 'LineWidth', 1.5), title('CANADA', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,5)
plot(t_2,mtx_z(:,5), 'LineWidth', 1.5), title('DENMARK', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,6)
plot(t_2,mtx_z(:,6), 'LineWidth', 1.5), title('NORWAY', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,7)
plot(t_2,mtx_z(:,7), 'LineWidth', 1.5), title('SWEDEN', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,8)
plot(t_2,mtx_z(:,8), 'LineWidth', 1.5), title('SWITZERLAND', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
subplot(3,3,9)
plot(t_2,mtx_z(:,9), 'LineWidth', 1.5), title('KOREA', 'FontSize', 11), ylabel('Index'), set(gca,'FontSize',11),  set(gcf, 'color',  'w');
sgtitle('Graph: Real Exchange Rate (1995m2-2007m12)')



%% 
%/////////////////////////////////%
%/////////--Question 1--//////////%
%/////////////////////////////////
seriesnam=["EUR","UK", "JAP", "CAN", "DEN", "NOR", "SWE", "SWI", "KOR"]';
countries_2=repelem(seriesnam,155)
countries_3=countries_2'
countries_4=countries_3'
count_5=countries_4
countr= categorical(count_5);
%---///Q1.A//---Pooled regression of all cointries--
y=s_pool;
x=z_pool;
mdl_pool=fitlm(x,y);
mdl_pool
a_pool=table2array(mdl_pool.Coefficients(1,1));
b_pool=table2array(mdl_pool.Coefficients(2,1));
t_pool_intercept=table2array(mdl_pool.Coefficients(1,3));
t_pool_coef=table2array(mdl_pool.Coefficients(2,3));
a_b_pool=[a_pool; b_pool];
t_pool_student= [t_pool_intercept; t_pool_coef]

%Plotting the results
 
y_graph = b_pool*x;
%f_1=figure;
f_1=figure;
%scatter(x,y,'filled');
gscatter(x,y,countr,[],'.',15);
hold on;
%plot(x,y_graph,'k-', 'LineWidth', 1.8);
plot(x,predict(mdl_pool),'k-', 'LineWidth', 2);
xlabel('Ln of Real Exchange Rate');
ylabel('Ln of Nominal Exchange Rate in US $ Term');
title('Graph: OLS, Pooled Data across Countries');
legend('boxoff');

%Ploting OLS residuals

figure,
xlabel('OLS Residuals')
boxplot(mdl_pool.Residuals.Raw,countr,'orientation','horizontal')
title('Graph: OLS Residuals of the Pooled Data across Countries');

%Results

disp('-------------------------------------------------------------------');
disp(' 1.(A). Pool data across countries and estimate the pooled variant ');
disp('-------------------------------------------------------------------');
mdl_pool

% https://uk.mathworks.com/matlabcentral/fileexchange/46515-multilevel-mixed-effects-modeling-using-matlab

clear y x 

%---///Q1.B//---Specifying the Countries regressions --- (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
figure; %This is basically the same graph as the one above 
sz=10
for i=1 : size(mtx_diff,2)
    y=mtx_diff(:,i);
    x=mtx_z(:,i);
    mdl_ctr=fitlm(x,y);
    a_ctr(:,i)=table2array(mdl_ctr.Coefficients(1,1));
    b_ctr(:,i)=table2array(mdl_ctr.Coefficients(2,1));
    t_ctr_intercept(:,i)=table2array(mdl_ctr.Coefficients(1,3));
    t_ctr_coef(:,i)=table2array(mdl_ctr.Coefficients(2,3));
    raw_res(:,i)=table2array(mdl_ctr.Residuals(:,1));
    y_graph = b_pool*x;
    
    scatter(x,y,sz,'filled');
    hold on;
    plot(x,y_graph,'k-', 'LineWidth', 2);
    xlabel('Ln of Real Exchange Rate');
    ylabel('Ln of Nominal Exchange Rate in US $ Term');
    title('Graph: OLS, Pooled Data across Countries');
    
end

%--Storing a and b (the constant and the coefficient) and the t-student of both for all countries (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
%(Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
a_b_ctr=[a_ctr; b_ctr];
t_student_ctr= [t_ctr_intercept; t_ctr_coef] 

a_b_ctr
t_student_ctr


Countries={'EUR', 'UK', 'JAP', 'CAN', 'DEN', 'NOR', 'SWE', 'SWI', 'KOR'}';
Intercept =(a_ctr)';
Coefficient=(b_ctr)';
t_Stat_inter=(t_ctr_intercept)';
t_Stat_Coeff=(t_ctr_coef)';
results_b=table(Countries, Intercept, t_Stat_inter, Coefficient,  t_Stat_Coeff  );
disp('-------------------------------------------------------------------');
disp(' 1.(B). Pool data across countries and estimate the pooled variant ');
disp('-------------------------------------------------------------------');






%---///Q1.C//-----Mean group estimator of alphas and betas 
mg_a_b=[mean(a_b_ctr(1,:)); mean(a_b_ctr(2,:))] ;
M_Intercept=[mean(a_b_ctr(1,:))];
M_Coefficient=[mean(a_b_ctr(2,:))];
M_t_Intercept=[mean(t_ctr_intercept)];
M_t_Coefficient=[mean(t_ctr_coef)];
mg={'Mean Group Estimator'}
results_c=table(mg,M_Intercept,M_t_Intercept,M_Coefficient,M_t_Coefficient);
disp('-------------------------------------------------------------------');
disp('            1.(C). Computing the mean group estimator              ');
disp('-------------------------------------------------------------------');
results_c

%////////////----Statistical tests----//////

%Heterosckedasricity and multicollinearity test for z
for i = 1:size(mtx_z,2);
    het_z(i,2)=archtest(mtx_z(:,i));
end

%archtest(res) returns a logical value with the rejection decision from conducting the Engles ARCH test for residual heteroscedasticity in the univariate residual series res.
% The result h = 1 indicates that you should reject null hypothesis of no conditional heteroscedasticity and conclude that there are significant ARCH effects in the return series.

%Durbin-Watson test p-values (Ho: Residuals from a linear regression are
%uncorrelated) and stationary 
for i=1:size(mtx_diff,2)
    p(1,i)=dwtest(raw_res(:,i),mtx_diff(:,i));
    p(2,i)=dwtest(raw_res(:,i),mtx_diff(:,i));
    %Stationary test
    stat(1,i) = kpsstest(mtx_diff(:,i));
    stat(2,i) = kpsstest(mtx_fx(:,i));
end

% dwtest(r,x) returns the p-value for the Durbin-Watson test of the null hypothesis that the residuals from a linear regression are uncorrelated. The alternative hypothesis is that there is autocorrelation among the residuals.

%kpsstest(y) returns the logical value (h) with the rejection decision from conducting the Kwiatkowski, Phillips, Schmidt, and Shin (KPSS) test for a unit root in the univariate time series y.
%--Test rejection decisions, returned as a logical value or vector of logical values with a length equal to the number of tests that the software conducts.
%---h = 1 indicates rejection of the trend-stationary null in favor of the unit root alternative.
%---h = 0 indicates failure to reject the trend-stationary null.

%Clearing information (In order to get a more space, and not get confuse,
%clear b_ctr a_ctr y_graph a_pool b_pool eur eur_1 unk unk_1 jap jap_1 can can_1 den den_1 nor nor_1 swe swe_1 swi swi_1 kor kor_1 uns uns_1

%/////////////////////////////////%
%/////////--Question 2--//////////%
%/////////////////////////////////

disp('-------------------------------------------------------------------');
disp(' 2. Estimating Within-Group Estimator (Fixed Effects Model)         ');
disp('-------------------------------------------------------------------');

number=[1 2 3 3 4 5 6 7 8 9]';
number={1 2 3 4 5 6 7 8 9}'
number_1=repelem(number,155)
number_2 = cell2mat(number_1)

figure;
boxplot(s_pool,countries_2)
xlabel('Countries'); 
title('Graph: Plot data by Countries');
%figure;
%boxplot(s_pool,year)
%xlabel('Year');
%title('Graph: Plot data by Year');

n_ctr=size(mtx_diff,2);
%Subtracting the average value of each country from the country observation
%for tha variable z and diff
for i=1 : n_ctr
    x_z(:,i)=mtx_z(:,i)-mean(mtx_z(:,i));
    y_s(:,i)=mtx_diff(:,i)-mean(mtx_diff(:,i));
end

%Reshaping again
y_s_1=reshape(y_s,ii,1); %s_pool_fe
x_z_1=reshape(x_z,ii,1); %z_pool_fe



%/////////////////////////////////////////////////////////////////////////////////

t=size(mtx_diff,1);
mtx=zeros(n_ctr*t,n_ctr);

%Matrix to calculate final coefficients
ii=1;
for i=1 :n_ctr
    mtx(ii:t*i,i)=ones(t,1);
    ii=t*i+1;
end

m=eye(n_ctr*t)-mtx*inv(mtx'*mtx)*mtx';

%-----
b_fe=regstats(y_s_1,x_z_1,'linear')
b_fe
b__fe=inv(x_z_1'*x_z_1)*(x_z_1'*y_s_1)
beta_fe=inv(x_z_1'*m*x_z_1)*x_z_1'*m*y_s_1;
%alphas_fe=inv(mtx'*mtx)*mtx'*(y_s_1-x_z_1*beta_fe);
%/////////////////////////////////////////////////////////////////////////////////



%Building a table 
tb_fe=table(y_s_1,x_z_1,number_2,'VariableNames',{'NominalEx','RealEx','Countries'} );
tb_fe

% Fixed Effect Estimation (Within Estimator)
mdl_fe=fitlm(x_z_1,y_s_1);
mdl_fe

mdl_tb_fe=fitlm(tb_fe,'NominalEx ~ 1 + RealEx ')

disp(['ExReal:',num2str(mdl_tb_fe.Coefficients{'RealEx',1})]);


%--Test the hypothesis that the constant term is the same for all cross
 %sectional units----Calculating SSQ for fixed effect
 
 ssq_fe=sum(b_fe.r.^2)/(n_ctr*t-n_ctr-1);
 
 %Calculating of RSS and SSQ of the countries
 i=size(raw_res,1)*size(raw_res,2);
 raw_res=reshape(raw_res,i,1);
 rss_ctr=sum(raw_res.^2);   
 ssq_ctr=rss_ctr/n_ctr*t;
 
 %--Calculating of the Maximum log-likelihood
 max_1_ctr=-0.5*n_ctr*t*(log(2*pi)+1)-0.5*n_ctr*t*log(ssq_ctr);

 
 %Calculating of the Maximum log-likelihood for the fixed effect
  max_1_fe=-0.5*n_ctr*t*(log(2*pi)+1)-0.5*n_ctr*t*log(ssq_fe);
 
 %Lagrange multipliers test
 lr_test=2*(max_1_ctr-max_1_fe);
 lr_crit=chi2inv(0.95,(n_ctr-1));
 lr_result=lr_test-lr_crit;
 
 clear ree_ctr ssq_ctr raw_res m rss_ctr

%/////////////////////////////////%
%/////////--Question 3--//////////%
%/////////////////////////////////
 
 %%%%%--Fixed Effect vs Random effect--%%%
 %-Calculating of theta
 res_pool=table2array(mdl_pool.Residuals(:,1));
 ssq_pool=sum(res_pool.^2)/(n_ctr*t-2);
 ssq_dif=ssq_pool-ssq_fe;
 theta=1-(ssq_fe/(t*ssq_dif+ssq_fe))^(1/2);
 
 %-Loop for the value of x for the Random effect
 for i=1:size(mtx_z,2)
     x_re(:,i)=mtx_z(:,1)-mean(mtx_z(:,i))*theta;
 end

 %Loop for the value of y for Random Effect
 for i=1:size(mtx_diff,2)
     y_re(:,i)=mtx_diff(:,i)-mean(mtx_diff(:,i))*theta;
 end

 %Size of the vector
 i=size(x_re,1)*size(x_re,2);
 ii=size(y_re,1)*size(y_re,2);
 y_re=reshape(y_re,ii,1);
 x_re=reshape(x_re,i,1);
 mdl_re=fitlm(x_re,y_re);
 mdl_re  % Here the coefficient of the slope is verely the same as the panel data toolbox
 
 %Hausman test
 b_re=table2array(mdl_re.Coefficients(2,1));
 q=beta_fe-b_re;
 re_cov=mdl_re.CoefficientCovariance(2,2);
 v_q=b_fe.covb(2,2)-re_cov;
 
 h_test=q'*inv(v_q)*q;
 h_cirt=chi2inv(0.95,1);
 h_result=h_test-h_cirt;
 
 res_re_1=table2array(mdl_re.Residuals(:,1));
 res_re=sum(res_re_1.^2);
 
 %------------------------------------------------------------------%%%
 %----------------- Panel Data Models using the Panel toolbox-------%%%
 %------------------------------------------------------------------%%%
 

ynames = {'FX'};
xnames = {'RFX'};

v=[1 2 3 4 5 6 7 8 9 ]' %This comand will create the identificator each countries (Euro UK JAP CAN DEN NORWAY SWE SWI KOR)
%v_1=[1 : n_ctr ]' % This comand creates the same but it can be re-adjust wherever for other type of data
id=repelem(v,155)   %Here the identificator will be fill into the data each 155 times

t_2=[1995.083:0.0833:2007.91667]'
year=[ t_2' t_2' t_2' t_2' t_2' t_2' t_2' t_2' t_2'  ]'
t_panel=repelem(t_2:t_2,9)


disp('------------------------------------------------------------------');
disp('    PANEL DATA MODELS (Countries Nominal Exchange Rate Effect)    ');
disp('------------------------------------------------------------------');
 

% OLS
regols = ols(s_pool,z_pool);
regols.ynames = ynames;
regols.xnames = xnames;
estdisp(regols);

% Clustered OLS
regolsc = ols(s_pool,z_pool,'vartype','cluster','clusterid',id);
regolsc.ynames = ynames;
regolsc.xnames = xnames;
estdisp(regolsc);

% Panel FE
regfe = panel(id,year,s_pool, z_pool, 'fe');
regfe.ynames = ynames;
regfe.xnames = xnames;
estdisp(regfe);

iregfe=ieffects(regfe);
ieffectsdisp(regfe);

iregfe_over=ieffects(regfe,'overall')
ieffectsdisp(regfe, 'overall');

% Panel BE
regbe = panel(id,year,s_pool, z_pool, 'be');
regbe.ynames = ynames;
regbe.xnames = xnames;
estdisp(regbe);

% Panel RE
regre = panel(id,year,s_pool, z_pool, 're');
regre.ynames = ynames;
regre.xnames = xnames;
estdisp(regre);
estcidisp(regre);

 
% F test of inividual effects
effF = effectsftest(regfe);
testdisp(effF);

% BP test for effects
bpre = bpretest(regre);
testdisp(bpre);

% Hausman test
h = hausmantest(regfe, regre);
testdisp(h);

% Mundlak test
mu = mundlakvatest(regfe);
testdisp(mu);

% Pool test
po = pooltest(regfe);
testdisp(po);

% Wooldridge serial test
wo = woolserialtest(regfe);
testdisp(wo);

wo = woolserialtest(regfe,'dfcorrection',0);
testdisp(wo);

% Baltagi Li serial
bl = blserialtest(regre);
testdisp(bl)

% Pesaran CSD
pecsdfe = pesarancsdtest(regfe);
testdisp(pecsdfe);

pecsdre = pesarancsdtest(regre);
testdisp(pecsdre);


% Panel FE Robust
regfer = panel(id, year, s_pool, z_pool, 'fe', 'vartype', 'robust');
regfer.ynames = ynames;
regfer.xnames = xnames;
estdisp(regfer);

% Panel RE Robust
regrer = panel(id, year, s_pool, z_pool, 're', 'vartype', 'robust');
regrer.ynames = ynames;
regrer.xnames = xnames;
estdisp(regrer);

% Mundlak test
mur = mundlakvatest(regfer);
testdisp(mur);

% Individual effects
ieffectsdisp(regfer);

% Individual effects
ieffectsdisp(regfer,'overall');


toc
















