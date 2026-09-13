from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from xgboost import XGBClassifier
BASE=Path(__file__).resolve().parent.parent; MD=BASE/"models"; SENSORS=["pH","Temp","Voltage"]; DISPLAY={"pH":("upper",2.20),"Temp":("lower",40.0),"Voltage":("lower",15.0)}
def _load(p):m=XGBClassifier();m.load_model(p);return m
FC=json.loads((MD/"final_config.json").read_text(encoding="utf-8"));FP=joblib.load(MD/"final_preprocess.joblib");FM=_load(MD/"final_defect_xgb.json")
EC=json.loads((MD/"early_config.json").read_text(encoding="utf-8"));EP=joblib.load(MD/"early_preprocess.joblib");EM=_load(MD/"early_warning_xgb.json")
def _std(x):
 x=x.copy().rename(columns={"datetime":"Timestamp","ph":"pH","temp":"Temp","voltage":"Voltage"});x["Timestamp"]=pd.to_datetime(x["Timestamp"]);return x.sort_values("Timestamp").reset_index(drop=True)
def _risk(prob,cal):
 cal=np.asarray(cal,float);med=np.median(cal);q1,q3=np.percentile(cal,[25,75]);sc=q3-q1
 if not np.isfinite(sc) or sc<1e-12:sc=np.std(cal,ddof=1)
 if not np.isfinite(sc) or sc<1e-12:sc=1.0
 return float((prob-med)/sc),float(np.searchsorted(np.sort(cal),prob,side="right")/max(len(cal),1))
def current_snapshot(lot_df):
 x=_std(lot_df);out={"measurement_count":len(x)}
 if len(x)>1:out["elapsed_sec"]=(x.Timestamp.iloc[-1]-x.Timestamp.iloc[0]).total_seconds()
 for s in SENSORS:
  v=x[s].to_numpy(float);direction,limit=DISPLAY[s];rate=float(np.mean(v>limit)*100 if direction=="upper" else np.mean(v<limit)*100);out[s]={"current":float(v[-1]),"std":float(np.std(v,ddof=1)) if len(v)>1 else 0.0,"min":float(v.min()),"IQR":float(np.percentile(v,75)-np.percentile(v,25)),"display_excursion_rate_pct":rate}
 return out
def predict_final_lot(lot_df):
 x=_std(lot_df);f={};ref=FC["excursion_reference"]
 for s in SENSORS:
  v=x[s].to_numpy(float);f[f"{s}_std"]=float(np.std(v,ddof=1));f[f"{s}_min"]=float(v.min());f[f"{s}_IQR"]=float(np.percentile(v,75)-np.percentile(v,25));f[f"{s}_exc_high_rate"]=float(np.mean(v>ref[s]["upper"]))
 z=pd.DataFrame([f])[FC["features"]].fillna(FP["median"]);prob=float(FM.predict_proba(FP["scaler"].transform(z.to_numpy(np.float32)))[:,1][0]);score,pct=_risk(prob,FC["calibration_scores"]);pred=prob>=FC["threshold"]
 return {"prediction":"불량 위험" if pred else "정상 위험","prediction_code":int(pred),"raw_probability":prob,"risk_score":score,"risk_percentile":pct,"threshold":float(FC["threshold"]),"validated_pipeline":True}
def predict_early_warning(lot_df):
 x=_std(lot_df);levels=[int(v) for v in EC["progress_levels"]];lengths={int(k):int(v) for k,v in EC["prefix_lengths"].items()};ok=[p for p in levels if lengths[p]<=len(x)]
 if not ok:return {"available":False,"status":"데이터 부족","validated":False}
 p=max(ok);x=x.iloc[:lengths[p]];ref=EC["progress_excursion_reference"][str(p)];f={"Progress_Ratio":p/100.0}
 for s in SENSORS:
  v=x[s].to_numpy(float);f[f"{s}_std"]=float(np.std(v,ddof=1));f[f"{s}_min"]=float(v.min());f[f"{s}_IQR"]=float(np.percentile(v,75)-np.percentile(v,25));f[f"{s}_exc_high_rate"]=float(np.mean(v>ref[s]["upper"]))
 z=pd.DataFrame([f])[EC["features"]].fillna(EP["median"]);prob=float(EM.predict_proba(EP["scaler"].transform(z.to_numpy(float)))[:,1][0]);score,pct=_risk(prob,EC["calibration_scores"][str(p)]);thr=float(EC["thresholds"][str(p)]);status="높음" if pct>=.97 else ("주의" if pct>=.90 else "낮음")
 return {"available":True,"checkpoint":p,"status":status,"raw_probability":prob,"risk_score":score,"risk_percentile":pct,"reference_threshold":thr,"threshold_alert":bool(prob>=thr),"validated":False,"display_note":"참고용 Early Warning 신호이며 최종 불량 판정이 아닙니다."}
