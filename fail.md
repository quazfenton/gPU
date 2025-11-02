----## FIX: give full corrected version of kaggle.py, my custom scriptstill errors, none work or run/open in kaggle then deploy from kaggle notebool to gcloud functions as api , which should all happen in the commmand NOT just pull. notebook should be opened via kaggle upload/run commands and from inside kaggle environment it should deploy itself as an api   ie. using an LLM model hosted on kaggle. notebook file should download (clone/wget/curl the .ipynb if from remote/git  OR pull if from kaggle.com)  from remote then (or) pushed from local/run from uploaded or aready existing kaggle remoet notebook,   ) secondly, some kaggle subdirectoried download successfully, as you can see from ls at the end,  from pullbut dont run after (maye wrong ath or name is sent).. (none of the remote files do download.   also,  change order of auth attempt fallback,  kaggleshub sdk last necause for some reason dotenv doesnt recognize .env in project root, maybe venv error


----##OUTPUT:" ls
deepseek-r1-0528-example-starter-notebook.ipynb  deploy_model.py  deploy.sh  doctor.py  fail.md  kaggle.py  __pycache__  README.markdown  requirements.txt  venv
[admin@compute kaggle]$ python kaggle.py list
python kaggle.py run bubbahubba/notebook73152bd999
python kaggle.py run https://github.com/odewahn/ipynb-examples/blob/master/Importing%20Notebooks.ipynb
python kaggle.py run
python kaggle.py create /home/admin/ai/kaggle/deepseek-r1-0528-example-starter-notebook.ipynb
python kaggle.py create chitradey/ba-using-python-11-cluster-analysis
python kaggle.py create 
python kaggle.py run /home/admin/ai/kaggle/deepseek-r1-0528-example-starter-notebook.ipynb
ls
python kaggle.run chitradey/ba-using-python-11-cluster-analysis
python kaggle.py create https://github.com/quazfenton/2067/blob/main/vision/getting-started/veo3_video_generation.ipynb
python kaggle.py run deepseek-r1-0528-example-starter-notebook.ipynb
ls

##ERRORS. ALL FAILED
2025-08-20 18:00:12,005 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:12,005 - INFO - Executing: kaggle kernels list
REFERENCE                                TITLE                                              LAST RUN                 
-------------------------------------------------------------------------------------------------------------------
--------------------------------------   ------------------------------------------------   ----------               
hopesb/binary-classification-with-bank   Binary Classification with Bank Dataset Shodolam   36                       
yusuketogashi/lb-0-598-mdc-fork-of-sol   LB 0.598 | MDC | Fork of Solution ed608a Yusuke    40                       
sandeepaashish/week4-workbook            week4-workbook Sandeep Aashish 2025-08-20 10:56:   22                       
shivaditya2004/22f3000819-ka3            22f3000819_KA3 Shivaditya2004 2025-08-20 15:10:5   7                        
chitradey/ba-using-python-11-cluster-a   BA Using Python 11: Cluster Analysis Chitra Dey    24                       
insmora/ps5e8-term-deposit-prediction-   PS5E8 - Term Deposit Prediction |CB, LGBM, XGB I   4                        
sayantanghsoh/23f3003980assignment-3     23f3003980Assignment 3 SAYANTAN GHSOH 2025-08-20   7                        
moustafamo/gym-data-faker                GYM data_faker MoustafaMo 2025-08-20 17:39:50.45   3                        
zkogan/cayleypy-teraminx-beam-search-g   CayleyPy <> Teraminx [Beam Search | GPU] Zakhar    4                        
hellosourav/cifar-10-on-msvmnet          CIFAR-10 on MSVMNet Sourav Das 2025-08-20 16:31:   1                        
sulimanabusamak123/breast-cancer-class   Breast Cancer Classification (Neural Network) su   2                        
daosyduyminh/simple-ensemble             Simple Ensemble Dao Sy Duy Minh 2025-08-20 06:23   25                       
hiranorm/offline-install-ver-fork-of-q   Offline install Ver|Fork of Qwen3+Qwen2.5+Llama3   11                       
patilswarup/house-price-prediction-ml    House Price Prediction|ML patil_swarup 2025-08-2   2                        
gauravduttakiit/dpc-good-quality-gpu     DPC : good_quality : gpu Gaurav Dutta 2025-08-20   3                        
bengj10/plant-disease-prediction-cnn-w   Plant Disease Prediction ~ CNN with explanation    3                        
masonlai42/experimental                  Experimental Mason Lai 42 2025-08-20 08:39:03.80   15                       
vines666/xbox-games-pass-portfolio-dat   Xbox Games Pass - Portfolio Data Analysis Rohan    2                        
sumedhamishra12/eatery-prediction        Eatery_Prediction SUMEDHA MISHRA 2025-08-20 17:5   1                        
gauravduttakiit/dpc-medium-quality-gpu   DPC : medium_quality : gpu Gaurav Dutta 2025-08-   3                        
2025-08-20 18:00:13,123 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:13,123 - INFO - Executing: kaggle kernels pull bubbahubba/notebook73152bd999 --path .
2025-08-20 18:00:13,729 - INFO - Kernel pulled to .
2025-08-20 18:00:13,729 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:13,729 - INFO - Executing: kaggle kernels status bubbahubba/notebook73152bd999
2025-08-20 18:00:14,239 - INFO - Kernel status: {'status': 'bubbahubba/notebook73152bd999 has status "KernelWorkerStatus.COMPLETE"'}
2025-08-20 18:00:14,923 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:14,924 - INFO - Created metadata for kernel: bubbahubba/tmpwyap4e4q-1755727214
2025-08-20 18:00:14,924 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:14,924 - INFO - Executing: kaggle kernels push --path /tmp/tmpfq3aflgc
2025-08-20 18:00:15,260 - ERROR - Command failed: Command '['kaggle', 'kernels', 'push', '--path', '/tmp/tmpfq3aflgc']' returned non-zero exit status 1.
2025-08-20 18:00:15,261 - ERROR - stdout: Your kernel title does not resolve to the specified id. This may result in surprising behavior. We suggest making your title something that resolves to the specified id. See https://en.wikipedia.org/wiki/Clean_URL#Slug for more information on how slugs are determined.
Expecting value: line 7 column 1 (char 6)

2025-08-20 18:00:15,261 - ERROR - stderr: 
2025-08-20 18:00:15,261 - ERROR - Push failed: 
2025-08-20 18:00:15,261 - ERROR - Push failed: Command '['kaggle', 'kernels', 'push', '--path', '/tmp/tmpfq3aflgc']' returned non-zero exit status 1.
2025-08-20 18:00:15,496 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:15,496 - INFO - Executing: kaggle kernels list

Available kernels:
1. -------------------------------------------------------------- (-------------------------------------------------- ----------------- --------------------------)
2. hopesb/binary-classification-with-bank-dataset (Binary Classification with Bank Dataset Shodolamu Opeyemi 2025-08-20 17:19:22.710000)
3. yusuketogashi/lb-0-598-mdc-fork-of-solution-ed608a (LB 0.598 | MDC | Fork of Solution ed608a Yusuke Togashi 2025-08-20 13:27:21.743000)
4. sandeepaashish/week4-workbook (week4-workbook Sandeep Aashish 2025-08-20 10:56:52.837000)
5. shivaditya2004/22f3000819-ka3 (22f3000819_KA3 Shivaditya2004 2025-08-20 15:10:55.433000)
6. chitradey/ba-using-python-11-cluster-analysis (BA Using Python 11: Cluster Analysis Chitra Dey 2025-08-20 08:16:13.520000)
7. insmora/ps5e8-term-deposit-prediction-cb-lgbm-xgb (PS5E8 - Term Deposit Prediction |CB, LGBM, XGB Inés Mora 2025-08-20 18:24:19.840000)
8. sayantanghsoh/23f3003980assignment-3 (23f3003980Assignment 3 SAYANTAN GHSOH 2025-08-20 16:20:29.933000)
9. moustafamo/gym-data-faker (GYM data_faker MoustafaMo 2025-08-20 17:39:50.453000)
10. zkogan/cayleypy-teraminx-beam-search-gpu (CayleyPy <> Teraminx [Beam Search | GPU] Zakhar Kogan 2025-08-20 21:29:28.687000)
11. hellosourav/cifar-10-on-msvmnet (CIFAR-10 on MSVMNet Sourav Das 2025-08-20 16:31:35.673000)
12. sulimanabusamak123/breast-cancer-classification-neural-network (Breast Cancer Classification (Neural Network) suliman abusamak 2025-08-20 16:22:58.707000)
13. daosyduyminh/simple-ensemble (Simple Ensemble Dao Sy Duy Minh 2025-08-20 06:23:48.223000)
14. hiranorm/offline-install-ver-fork-of-qwen3-qwen2-5-llama3-1 (Offline install Ver|Fork of Qwen3+Qwen2.5+Llama3.1 Hira Norm 2025-08-20 11:10:50.713000)
15. patilswarup/house-price-prediction-ml (House Price Prediction|ML patil_swarup 2025-08-20 18:11:35.637000)

Choose kernel number (or 'q' to quit): 5
2025-08-20 18:00:35,909 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:35,909 - INFO - Executing: kaggle kernels pull shivaditya2004/22f3000819-ka3 --path .
2025-08-20 18:00:36,943 - INFO - Kernel pulled to .
2025-08-20 18:00:36,943 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:36,944 - INFO - Executing: kaggle kernels status shivaditya2004/22f3000819-ka3
2025-08-20 18:00:37,399 - INFO - Kernel status: {'status': 'shivaditya2004/22f3000819-ka3 has status "KernelWorkerStatus.COMPLETE"'}
2025-08-20 18:00:37,634 - ERROR - Command failed: 'Namespace' object has no attribute 'path'
2025-08-20 18:00:37,853 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:37,853 - INFO - Created metadata for kernel: bubbahubba/notebook-1755727237
2025-08-20 18:00:37,853 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:37,853 - INFO - Executing: kaggle kernels push --path /tmp/tmpu8lv4b0c
2025-08-20 18:00:38,361 - ERROR - Command failed: Command '['kaggle', 'kernels', 'push', '--path', '/tmp/tmpu8lv4b0c']' returned non-zero exit status 1.
2025-08-20 18:00:38,361 - ERROR - stdout: Your kernel title does not resolve to the specified id. This may result in surprising behavior. We suggest making your title something that resolves to the specified id. See https://en.wikipedia.org/wiki/Clean_URL#Slug for more information on how slugs are determined.
409 Client Error: Conflict for url: https://www.kaggle.com/api/v1/kernels/push

2025-08-20 18:00:38,361 - ERROR - stderr: 
2025-08-20 18:00:38,361 - ERROR - Push failed: 
2025-08-20 18:00:38,361 - ERROR - Failed to create notebook: Command '['kaggle', 'kernels', 'push', '--path', '/tmp/tmpu8lv4b0c']' returned non-zero exit status 1.
2025-08-20 18:00:38,576 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:38,576 - INFO - Created metadata for kernel: bubbahubba/notebook-1755727238
2025-08-20 18:00:38,576 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:38,576 - INFO - Executing: kaggle kernels push --path /tmp/tmpwluvyaco
2025-08-20 18:00:39,080 - ERROR - Command failed: Command '['kaggle', 'kernels', 'push', '--path', '/tmp/tmpwluvyaco']' returned non-zero exit status 1.
2025-08-20 18:00:39,080 - ERROR - stdout: Your kernel title does not resolve to the specified id. This may result in surprising behavior. We suggest making your title something that resolves to the specified id. See https://en.wikipedia.org/wiki/Clean_URL#Slug for more information on how slugs are determined.
409 Client Error: Conflict for url: https://www.kaggle.com/api/v1/kernels/push

2025-08-20 18:00:39,080 - ERROR - stderr: 
2025-08-20 18:00:39,080 - ERROR - Push failed: 
2025-08-20 18:00:39,080 - ERROR - Failed to create notebook: Command '['kaggle', 'kernels', 'push', '--path', '/tmp/tmpwluvyaco']' returned non-zero exit status 1.
2025-08-20 18:00:39,298 - ERROR - Command failed: 'Namespace' object has no attribute 'path'
22f3000819-ka3.ipynb                             deploy_model.py  doctor.py  kaggle.py                 __pycache__      requirements.txt
deepseek-r1-0528-example-starter-notebook.ipynb  deploy.sh        fail.md    notebook73152bd999.ipynb  README.markdown  venv
python: can't open file '/home/admin/ai/kaggle/kaggle.run': [Errno 2] No such file or directory
2025-08-20 18:00:39,993 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:39,994 - INFO - Created metadata for kernel: bubbahubba/tmp90j1p51u-1755727239
2025-08-20 18:00:39,994 - INFO - Using credentials from ~/.kaggle/kaggle.json
2025-08-20 18:00:39,994 - INFO - Executing: kaggle kernels push --path /tmp/tmpg378deqk
2025-08-20 18:00:40,329 - ERROR - Command failed: Command '['kaggle', 'kernels', 'push', '--path', '/tmp/tmpg378deqk']' returned non-zero exit status 1.
2025-08-20 18:00:40,329 - ERROR - stdout: Your kernel title does not resolve to the specified id. This may result in surprising behavior. We suggest making your title something that resolves to the specified id. See https://en.wikipedia.org/wiki/Clean_URL#Slug for more information on how slugs are determined.
Expecting value: line 7 column 1 (char 6)

2025-08-20 18:00:40,329 - ERROR - stderr: 
2025-08-20 18:00:40,329 - ERROR - Push failed: 
2025-08-20 18:00:40,330 - ERROR - Push failed: Command '['kaggle', 'kernels', 'push', '--path', '/tmp/tmpg378deqk']' returned non-zero exit status 1.
2025-08-20 18:00:40,567 - ERROR - Command failed: 'Namespace' object has no attribute 'path'
ls
22f3000819-ka3.ipynb                             deploy_model.py  doctor.py  kaggle.py                 __pycache__      requirements.txt
deepseek-r1-0528-example-starter-notebook.ipynb  deploy.sh        fail.md    notebook73152bd999.ipynb  README.markdown  venv


 "
-------

	[[[ Correct: should  automatic deploy from kaggle AUTOMATICALLY AFTER REMOTEDOWNLOAD/  OR LOCAl UPLOAD  of notebook -> running notebook in KAGGLE successful  should DEPLOY FROM THERE
gcloud functions deploy predict-handler-v2
--runtime python311
--trigger-http
--allow-unauthenticated
--memory 512MB
--timeout 3600s
--region us-central1

./deploy.sh ]]]
