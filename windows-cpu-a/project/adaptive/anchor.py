"""Pre-feedback center anchoring and a prequential, past-feedback-only gate."""
import numpy as np
from .base import CONFIG,predict
from .acquisition_v1 import search_centers


class AnchorContext:
    def __init__(self,models,x,y):
        self.models=models;self.x=x;self.y=y
        self.ids=search_centers(x,y)
        self.center_prediction=predict(models,x[self.ids])

    def shadow(self,points,raw=None):
        raw=predict(self.models,points) if raw is None else np.asarray(raw)
        if raw.shape!=(5,len(points)):
            raise RuntimeError('Expected five pre-feedback model predictions per candidate')
        distances=np.linalg.norm(points[:,None]-self.x[self.ids][None],axis=2)
        nearest=distances.argmin(1);distance=distances[np.arange(len(points)),nearest]
        indices=self.ids[nearest]
        eligible=distance<=CONFIG['anchor']['radius']
        anchored=self.y[indices][None]+raw-self.center_prediction[:,nearest]
        anchored[:,~eligible]=raw[:,~eligible]
        return dict(eligible=eligible.tolist(),center_indices=indices.tolist(),center_R=self.y[indices].tolist(),
            center_distance=distance.tolist(),raw_mean=raw.mean(0).tolist(),anchor_mean=anchored.mean(0).tolist(),
            raw_score=(raw.mean(0)+.5*raw.std(0)).tolist(),
            anchor_score=(anchored.mean(0)+.5*anchored.std(0)).tolist(),
            raw_predictions=raw.T.tolist(),anchor_predictions=anchored.T.tolist(),
            available_rows=len(self.y),available_online_rows=len(self.y)-2048,
            generated_before_feedback=True)

    def rank(self,points,raw=None):
        return np.asarray(self.shadow(points,raw)['anchor_score'])


def feedback_metrics(shadow,scores,round_number):
    scores=np.asarray(scores,dtype=float);eligible=np.asarray(shadow['eligible'],dtype=bool)
    if len(scores)!=len(eligible):raise RuntimeError('Shadow/feedback pairing mismatch')
    n=int(eligible.sum());raw=np.asarray(shadow['raw_mean']);anchored=np.asarray(shadow['anchor_mean'])
    report=dict(round=round_number,eligible_count=n,
                raw_abs_error_sum=float(np.abs(raw[eligible]-scores[eligible]).sum()),
                anchor_abs_error_sum=float(np.abs(anchored[eligible]-scores[eligible]).sum()),
                raw_top5_regret=None,anchor_top5_regret=None)
    if n>=5:
        ids=np.flatnonzero(eligible);best=float(scores[ids].min())
        for prefix in ('raw','anchor'):
            order=ids[np.argsort(np.asarray(shadow[prefix+'_score'])[ids],kind='stable')[:5]]
            report[prefix+'_top5_regret']=float(scores[order].min()-best)
    return report


def gate(history):
    valid=[v for v in history if v['eligible_count']>=5 and v['raw_top5_regret'] is not None]
    recent=valid[-3:];n=sum(v['eligible_count'] for v in recent)
    if len(recent)<3 or n<30:
        return dict(passed=False,reason='insufficient_past_feedback',valid_batches=len(recent),eligible_points=n)
    raw=sum(v['raw_abs_error_sum'] for v in recent)/n
    anchored=sum(v['anchor_abs_error_sum'] for v in recent)/n
    raw_regret=float(np.mean([v['raw_top5_regret'] for v in recent]))
    anchored_regret=float(np.mean([v['anchor_top5_regret'] for v in recent]))
    passed=raw>1e-12 and anchored<=.8*raw and anchored_regret<=raw_regret+1e-12
    return dict(passed=bool(passed),reason='passed' if passed else 'error_or_selection_guard_failed',
                valid_batches=3,eligible_points=n,raw_MAE=raw,anchor_MAE=anchored,
                raw_top5_regret=raw_regret,anchor_top5_regret=anchored_regret,
                assessed_rounds=[v['round'] for v in recent])


def update_feedback(state,batch,scores,previous_best,new_best):
    improvement=max(0.,float(previous_best-new_best))
    state['polish_stall_streak']=state.get('polish_stall_streak',0)+1 if improvement<.02 else 0
    active=bool(state.get('polish_active',False))
    if not active and batch['round']*50>=500 and state['polish_stall_streak']>=2:
        active=True;state['polish_trigger_round']=batch['round']
    state['polish_active']=active
    metric=feedback_metrics(batch['shadow'],scores,batch['round'])
    history=[v for v in state.get('anchor_history',[]) if v['round']!=batch['round']]
    history.append(metric);history.sort(key=lambda v:v['round']);state['anchor_history']=history
    decision=gate(history)
    state['anchor_enabled']=bool(active and decision['passed'])
    state['anchor_gate']=decision
    return dict(polish_active_next=active,polish_trigger_round=state.get('polish_trigger_round'),
                polish_stall_streak=state['polish_stall_streak'],anchor_enabled_next=state['anchor_enabled'],
                shadow_feedback=metric,anchor_gate=decision)
