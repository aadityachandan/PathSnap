"""Generate a report from recorded experiment outputs; never invent metrics."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('work/mpl').resolve()))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from pathsnap.data import load_trip
from pathsnap.decoder import geometry

DESCRIPTIONS={
'logistic':('Linear/probabilistic','A scaled linear baseline tests whether geometry features separate correct and incorrect candidates. Fast and interpretable, but cannot directly learn complex interactions.','Binary logistic loss; standardized features.'),
'gaussian_nb':('Probabilistic','A low-cost probabilistic baseline. Conditional independence is unlikely for distance, rank and distance margin, so probability quality requires scrutiny.','Gaussian class-conditional likelihood with variance smoothing.'),
'decision_tree':('Tree','Nonlinear thresholds capture geometry interactions and provide an interpretable tree. A single tree may overfit trip-specific patterns.','Gini impurity; depth and minimum leaf size regularize complexity.'),
'random_forest':('Ensemble','Averaging randomized trees captures nonlinear interactions and reduces single-tree variance. Larger models increase inference and artifact cost.','Bootstrap ensemble of Gini-impurity decision trees.'),
'hist_gradient_boosting':('Boosting','Sequential corrections capture nonlinear geometry patterns with binned features. Capacity and learning rate control bias/variance.','Binary log loss; histogram-based gradient boosting.'),
'mlp':('Neural network','A two-hidden-layer perceptron (64 and 32 ReLU units) learns smooth nonlinear combinations of the scaled features. It needs feature scaling, is sensitive to learning rate and regularisation, and is harder to interpret.','Binary cross-entropy (log loss) minimised with Adam; early stopping on an internal validation split.')}


def plots(root,inventory,comparison,decoder,data_root):
    out=root/'results'/'plots';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    for metric,title in [('cv_top1_accuracy','Whole-trip CV: Top-1 accuracy (selection metric)'),('cv_macro_f1','Whole-trip CV: candidate Macro F1'),('top1_accuracy','Student test: derived-reference Top-1'),('macro_f1','Student test: candidate Macro F1'),('log_loss','Student test: candidate log loss')]:
        fig,ax=plt.subplots(figsize=(8,3.4));ax.barh(comparison.model,comparison[metric],color='#138b78');ax.set_title(title);ax.set_xlabel(metric);fig.tight_layout();fig.savefig(out/f'{metric}.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.2));axes[0].hist(inventory.gps_points,bins=20,color='#2779b4');axes[0].set(xlabel='GPS points per trip',ylabel='Trips');axes[1].scatter(inventory.gps_points,inventory.candidate_miss_rate,c=np.where(inventory.split=='train',0,1),cmap='coolwarm');axes[1].set(xlabel='GPS points',ylabel='Derived candidate miss rate');fig.tight_layout();fig.savefig(out/'eda.png');plt.close(fig)
    pivot=decoder.pivot(index='trip',columns='mode',values='top1_accuracy');ax=pivot.plot.bar(figsize=(8,3.5),color=['#2779b4','#138b78']);ax.set(ylabel='Derived-reference Top-1',title='Classifier versus HMM on each student-test trip');plt.xticks(rotation=45);plt.tight_layout();plt.savefig(out/'decoder.png');plt.close()
    results=json.loads((root/'artifacts/full_results.json').read_text())
    for r in results:
        fig,ax=plt.subplots(figsize=(4,3));cm=np.array(r['confusion_matrix']);ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted candidate class',ylabel='Derived candidate class',title=r['model']);fig.tight_layout();fig.savefig(out/(r['model']+'_confusion.png'));plt.close(fig)
    trip_id='00000052';tr=load_trip(Path(data_root)/'student_test'/trip_id)
    pred=json.loads((root/'results/predictions'/f'{trip_id}.json').read_text());reference=np.loadtxt(Path(data_root)/'student_test'/trip_id/f'{trip_id}.route',dtype=int,ndmin=1)
    fig,ax=plt.subplots(figsize=(8,4.5));a=tr.xy[tr.arcs[:,0]];b=tr.xy[tr.arcs[:,1]];lo=tr.gps.min(0)-100;hi=tr.gps.max(0)+100
    mask=np.all(np.maximum(a,b)>=lo,axis=1)&np.all(np.minimum(a,b)<=hi,axis=1)
    ax.add_collection(LineCollection(np.stack([a[mask],b[mask]],axis=1),colors='#cad2dc',linewidths=.6))
    ax.add_collection(LineCollection(np.stack([a[reference],b[reference]],axis=1),colors='#eba63b',linewidths=4,label='Reference'))
    ids=np.array([i for piece in pred['route_pieces'] for i in piece]);ax.add_collection(LineCollection(np.stack([a[ids],b[ids]],axis=1),colors='#138b78',linewidths=2,label='Prediction'))
    ax.scatter(tr.gps[:,0],tr.gps[:,1],s=4,color='#2779b4',label='GPS');ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Local east (m)',ylabel='Local north (m)',title=f'Student-test trip {trip_id}');ax.set_aspect('equal');ax.legend();fig.tight_layout();fig.savefig(out/'route_geometry.png');plt.close(fig)


def build(data_root):
    root=Path(__file__).resolve().parents[1];a=root/'artifacts';r=root/'results';out=root/'report';out.mkdir(exist_ok=True)
    inventory_path=root/'work/cache/inventory.csv'
    if not inventory_path.exists():inventory_path=r/'inventory.csv'
    inventory=pd.read_csv(inventory_path,dtype={'trip':str});inventory.to_csv(r/'inventory.csv',index=False)
    comparison=pd.read_csv(a/'model_comparison.csv');decoder=pd.read_csv(r/'decoder_comparison.csv',dtype={'trip':str});meta=json.loads((a/'metadata.json').read_text());configs=json.loads((a/'model_configs.json').read_text())
    plots(root,inventory,comparison,decoder,data_root)
    styles=getSampleStyleSheet();styles['Title'].textColor=colors.HexColor('#123e58');styles['Heading1'].textColor=colors.HexColor('#127d70');styles['BodyText'].leading=14
    story=[]
    def para(text,style='BodyText'):
        block=Paragraph(text,styles[style])
        if style=='BodyText' and len(story)>=2 and isinstance(story[-2],Paragraph) and story[-2].style.name.startswith('Heading'):
            group=story[-2:]+[block,Spacer(1,8)];del story[-2:];story.append(KeepTogether(group))
        else:
            story.append(KeepTogether([block]) if style=='BodyText' else block);story.append(Spacer(1,8))
    def heading(text):para(text,'Heading1')
    def pic(path,width=470,maxheight=310):
        from PIL import Image as PILImage
        w,h=PILImage.open(path).size;scale=min(width/w,maxheight/h);story.append(Image(str(path),width=w*scale,height=h*scale));story.append(Spacer(1,10))
    def table(headers,rows,widths=None):
        cells=[[Paragraph(str(x),styles['BodyText']) for x in row] for row in [headers]+rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeae9')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cbd5df')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(KeepTogether([t]));story.append(Spacer(1,12))
    para('PathSnap','Title');para('Map matching and route reconstruction','Heading2')
    team=out/'team.txt';para(team.read_text().replace('\n','<br/>') if team.exists() else 'Team names and IDs: to be completed by the team.')
    para('Development experiment report - generated from saved outputs. The two-laptop demonstration and reserved evaluator testing remain to be performed by the team. AI-assisted implementation and experimentation; review and disclose according to event policy.')
    para(f'Selected model: <b>{meta["selected_model"]}</b>. All metrics below are actual recorded development measurements. Point labels are derived from supplied reference-route alignment, not independently provided per-GPS truth.')
    heading('1. Dataset understanding and EDA')
    rows=[]
    for split,g in inventory.groupby('split'):rows.append([split,len(g),int(g.gps_points.sum()),int(g.missing_values.sum()),int(g.gaps_over_5s.sum())])
    table(['Partition','Trips','GPS points','Missing','Gaps >5s'],rows,[130,50,100,80,110])
    para('The raw files are whitespace-separated: track = longitude/latitude/time; nodes = longitude/latitude; arcs = directed from/to node indices; route = chronological arc indices. Each graph has local IDs. Evaluator data is absent from this experiment. Raw inputs are hashed in the inventory.')
    para('Validation rejects non-finite values, invalid indices and non-increasing times rather than silently dropping observations. Coordinates use a per-trip local equirectangular projection in metres. Inspect large geographical extents for distortion. Zero-length roads receive epsilon graph weights.')
    para(f'Training trip sizes range from {inventory[inventory.split=="train"].gps_points.min():,} to {inventory[inventory.split=="train"].gps_points.max():,} points. Weak graph components range from {inventory.weak_components.min():,} to {inventory.weak_components.max():,}. Mean per-trip derived-label distance coverage within 50m is {inventory.alignment_within_50m.mean():.1%}; see inventory.csv for each trip and sampling gaps.')
    para('Two training outliers deserve explicit review. For trip 00000054, only 12.80% of GPS observations are within 50m of ANY supplied reference segment; for trip 00000033 the figure is 41.20%. Thus disagreement is not solely caused by monotone ordering. The cause of incomplete agreement is unverified; unreliable aligned labels are filtered from training. See data_quality_notes.json.')
    pic(r/'plots/eda.png')
    heading('2. Features and reference alignment')
    para('For each GPS observation, the six nearest straight directed road segments are selected. A KD-tree indexes road samples at spacing at most 40m; radius expansion by 20m followed by exact segment projection retains exact nearest-segment ranking. Per-point features: distance, heading cosine, segment length, projection fraction, speed, time gap, distance rank, and distance above the nearest candidate.')
    cw=meta.get('context_windows',{'heading':10,'context':10})
    para(f'Error analysis of the first classifier on student-test data showed that most mistakes were (a) choosing the opposite-direction twin of the correct road, because a one-second heading is noisy, and (b) choosing an adjacent segment at segment boundaries. Six sequence-context features were therefore added (pathsnap/context.py): heading cosine over a +/-{cw["heading"]} point window; the fraction of the +/-{cw["context"]} neighbouring points that also have the same road as a candidate; their mean distance to it; the fraction of neighbours for which it is the nearest road; and whether the road equals or directly connects to a candidate of the previous / next point. They use only track, nodes and arcs. Window sizes were chosen by five-fold whole-trip CV on training trips only (artifacts/context_window_sweep.csv). Total: {len(meta["features"])} features.')
    para('The .route sequence is not a per-observation label array. A separate monotone dynamic program aligns observations to nondecreasing positions in the reference route. Cost: squared distance/20m plus heading penalty 2(1-cosine), with 0.1 per forward index jump. Start/end positions are free. This generates proxy labels and can make mistakes at repeated arcs or gaps. Reference information never enters candidate features or inference.')
    para(f'Classifier training uses at most {meta["training_observation_cap_per_trip"]} reproducibly sampled eligible observations per training trip ({meta["training_pairs"]:,} candidate pairs). Eligible points have aligned distance at most 50m and the aligned road in the candidate set. All student-test observations are retained for evaluation, including candidate misses. This sampling is a compute tradeoff, not full-observation training.')
    heading('3. Model experiments and selection')
    para(f'All {len(DESCRIPTIONS)} models use the same sampled candidate pairs and {meta.get("cv_folds",5)} whole-trip GroupKFold splits. Scaling is fitted inside each fold for scaled models. Primary selection metric: mean CV Top-1 accuracy, i.e. the fraction of GPS observations whose correct road receives the highest score, because the decoder needs the correct road ranked first; candidate Macro F1 at a 0.5 threshold does not measure this ranking. CV Macro F1 and log loss are reported as well. All student-test measurements are development comparisons. Candidate Macro F1 averages correct/incorrect classes and is not multiclass arc-ID F1. Top-1 is per GPS observation; log loss is per candidate pair.')
    table(['Model','CV Top-1','CV F1','Test Top-1','Test F1','Log loss'],[[row.model,f'{row.cv_top1_accuracy:.4f}',f'{row.cv_macro_f1:.4f}',f'{row.top1_accuracy:.4f}',f'{row.macro_f1:.4f}',f'{row.log_loss:.4f}'] for row in comparison.itertuples()],[140,66,62,66,62,62])
    for name in ['cv_top1_accuracy','cv_macro_f1','top1_accuracy','macro_f1','log_loss']:pic(r/'plots'/f'{name}.png')
    for name,(family,reason,loss) in DESCRIPTIONS.items():
        row=comparison[comparison.model==name].iloc[0];heading(name.replace('_',' ').title())
        para(f'Family: {family}. {reason} {loss}')
        important={k:v for k,v in configs[name].items() if k in ['max_depth','min_samples_leaf','n_estimators','max_iter','learning_rate','max_leaf_nodes','l2_regularization','logisticregression__C','gaussiannb__var_smoothing','mlpclassifier__hidden_layer_sizes','mlpclassifier__alpha','mlpclassifier__learning_rate_init','mlpclassifier__max_iter']}
        para('Initial configuration: '+', '.join(f'{k}={v}' for k,v in important.items())+'. Full parameters are in model_configs.json.')
        para(f'Measured CV Top-1: {row.cv_top1_accuracy:.4f} (fold std {row.cv_top1_std:.4f}); CV Macro F1: {row.cv_macro_f1:.4f}; student-test Macro F1: {row.macro_f1:.4f}; Top-1: {row.top1_accuracy:.4f}; log loss: {row.log_loss:.4f}. Precision/recall (macro): {row.macro_precision:.4f}/{row.macro_recall:.4f}. Fit time: {row.fit_seconds:.2f}s; candidate scoring time across all student-test trips: {row.inference_seconds:.2f}s (excludes feature generation and decoding).')
        para(f'Test minus CV F1 is {row.macro_f1-row.cv_macro_f1:+.4f}. This gap alone does not prove overfitting because CV uses filtered, sampled training observations and student testing uses all observations. Per-fold and per-trip outputs must be considered.'+(' No artificial epoch curves are generated for this model because it has no epoch-wise training history.' if name!='mlp' else ''))
        if name=='mlp' and (r/'plots/mlp_loss_curve.png').exists():
            hist=pd.read_csv(a/'mlp_history.csv');best=hist.loc[hist.validation_log_loss.idxmin()]
            para(f'Epoch-wise training and validation log loss were recorded (artifacts/mlp_history.csv) on a whole-trip hold-out. Validation loss is lowest at epoch {int(best.epoch)} ({best.validation_log_loss:.4f}); final epoch training/validation loss: {hist.train_log_loss.iloc[-1]:.4f}/{hist.validation_log_loss.iloc[-1]:.4f}. A growing gap would indicate overfitting.')
            pic(r/'plots/mlp_loss_curve.png',maxheight=230)
        pic(r/'plots'/f'{name}_confusion.png',maxheight=220)
    heading('4. Bayesian optimisation')
    para(f'Tuned family: {meta["tuned_family"]}. Optuna TPE ran {meta["trials"]} trials, with {meta["startup_trials"]} startup trials and subsequent model-guided proposals. Objective: {meta["selection_objective"]}, maximized. Search runtime: {meta["tuning_seconds"]:.1f}s. The search spaces are defined in member2_models/train.py; all bounds, trial values and fold objectives are retained in bayesian_trials.csv and source.')
    spaces={'hist_gradient_boosting':'max_leaf_nodes 7-31; learning_rate 0.03-0.2 log; l2_regularization 0.01-10 log; max_iter 60-140.','random_forest':'n_estimators 50-120; max_depth 8-22; min_samples_leaf 3-25.','decision_tree':'max_depth 5-22; min_samples_leaf 5-60.','logistic':'C 0.001-100, log scale.','gaussian_nb':'var_smoothing 1e-10 to 1e-2, log scale.'}
    para('Search space: '+meta.get('search_space',spaces.get(meta['tuned_family'],''))+'. Bounds target modest laptop-scale complexity while exploring regularization. Best parameters: '+str(meta['best_parameters']))
    baseline=comparison[comparison.model==meta['tuned_family']].iloc[0];tuned=comparison[comparison.model==meta['tuned_family']+'_tuned'].iloc[0]
    pic(r/'plots/bayesian_search.png',maxheight=230) if (r/'plots/bayesian_search.png').exists() else None
    para(f'Tuning changed CV Top-1 by {tuned.cv_top1_accuracy-baseline.cv_top1_accuracy:+.4f}, CV F1 by {tuned.cv_macro_f1-baseline.cv_macro_f1:+.4f}, student-test Top-1 by {tuned.top1_accuracy-baseline.top1_accuracy:+.4f}, student-test F1 by {tuned.macro_f1-baseline.macro_f1:+.4f}, and log loss by {tuned.log_loss-baseline.log_loss:+.4f}. The full comparison table includes both rows. The baseline is retained if tuning fails to improve the selection objective.')
    heading('5. HMM and Viterbi route reconstruction')
    para('Hidden states are candidate directed road segments at each observation. Emissions are normalized positive classifier scores with clipping before logarithms. Transitions use directed shortest-path movement between projected positions: log weight = -abs(network distance - observed displacement)/beta. Movements above 65m/s are invalid. Searches are bounded to 250-5000m based on observed displacement.')
    para(f'Decoder beta was selected from 10, 20, 35, 70, 140 and 350m using mean per-trip derived Top-1 on the ten student-test development trips. Selected beta: {meta["decoder"]["beta"]}m. A backward tolerance was also tuned (0-40m): GPS noise can make the projected position slip backwards on the same road between seconds; the first decoder version treated slips larger than 2% of the road length as impossible, which caused false route breaks and made the HMM worse than the classifier. Selected tolerance: {meta["decoder"].get("backward_tolerance",0)}m. These are development estimates, not untouched test estimates. Search records are in results/decoder_trials.csv.')
    para('Viterbi recurrence: V[t,j] = log emission[t,j] + max_i(V[t-1,i] + log transition[i,j]). Backpointers recover one road per observation. Missing/unreachable transitions or time gaps above 30s restart decoding explicitly. Reconstructed road pieces include shortest-path intermediate arcs; disconnected sections are not joined by fictitious lines. The remaining decoder limits are fixed engineering defaults.')
    summary=[]
    for mode,g in decoder.groupby('mode'):
        summary.append([mode,*[f'{np.average(g[col],weights=g.observations):.4f}' for col in ['top1_accuracy','strict_adjacency','route_arc_set_f1','sequence_similarity']]])
    table(['Mode','Top-1','Adjacency','Arc-set F1','Sequence'],summary,[110,90,90,90,90]);pic(r/'plots/decoder.png')
    pivot=decoder.pivot(index='trip',columns='mode',values='top1_accuracy');delta=pivot.hmm-pivot.classifier
    para(f'HMM improves derived Top-1 on {int((delta>0).sum())} trips, reduces it on {int((delta<0).sum())}, and ties on {int((delta==0).sum())}. {'The largest regression' if delta.min()<0 else 'The smallest change'} is {delta.min():+.4f} on trip {delta.idxmin()}; the largest gain is {delta.max():+.4f} on trip {delta.idxmax()}. Total decoder runtime is {decoder[decoder["mode"]=="hmm"].decoder_seconds.sum():.1f}s. These results do not imply universal HMM improvement.')
    para('Strict adjacency counts identical or directly connected consecutive assigned segments and may penalize legitimate movement across multiple edges. Arc-set F1 ignores order. Sequence similarity is SequenceMatcher matched-block ratio on a collapsed classifier sequence or HMM reconstructed arc sequence, compared with the supplied route. Candidate misses and proxy-label errors limit accuracy.')
    heading('6. Geographical visualisation')
    para('Interactive Folium maps for all ten student-test trips are in results/maps. Layers include observed GPS, nearby road geometry, reference route and prediction. Reference layers are only in development reports; evaluation UI maps exclude them. Road display is limited to the first 12,000 nearby edges for browser performance. Coordinates and all predictions come from actual data.')
    pic(r/'plots/route_geometry.png')
    screenshot=r/'folium_screenshot.jpg'
    if screenshot.exists():pic(screenshot,maxheight=320);para('Rendered Folium view of the actual development map.')
    else:para('Folium screenshot pending: open results/maps/00000052.html, capture the rendered map and save results/folium_screenshot.png, then regenerate. The figure above is a static geometry plot, not a Folium screenshot.')
    heading('7. API, UI and deployment')
    para('Laptop 1 runs FastAPI with the saved model, graph decoder and a separate post-prediction reference scorer. Laptop 2 runs the static interface and submits a trip ID. GET /trips lists configured trips; POST /match computes predictions, scalar metrics and a reference-free map. GET /health reports readiness. API logs record trip, runtime and break count. All ten available student-test trips are used for local checks when recorded in VALIDATION.md.')
    for file in sorted((r/'evidence').glob('*.jpg')) if (r/'evidence').exists() else []:pic(file);para('Local validation evidence: '+file.name+'. This is not evidence of two separate physical laptops.')
    evidence=out/'evidence';files=list(evidence.glob('*.png')) if evidence.exists() else []
    if files:
        for file in sorted(files):pic(file);para('Team-supplied deployment evidence: '+file.name)
    else:para('<b>Required team action:</b> add actual two-laptop UI and receiving-terminal screenshots to report/evidence and regenerate this PDF. Local API/UI checks do not substitute for this demonstration. The ten reserved evaluator trips were not available and have not been tested.')
    heading('8. Reproducibility, integrity and limitations')
    para('Run python run_all.py --data-root PATH to reproduce preparation, model comparisons, Bayesian trials, decoding, maps and this report. Exact seeds, sampled indices, folds, model parameters, raw hashes, dependencies, trial outcomes and per-trip metrics are included. The supplied results use the provided training and student-testing partitions only.')
    para('AI assistance was used in code generation, testing and report generation. Team members must review, understand, reproduce and disclose assistance as required. No evaluator accuracy, GitHub publication, two-laptop screenshot or experimental result is fabricated. Model artifacts are not evidence of independent student understanding.')
    para('Known limitations: proxy labels depend on alignment assumptions; capped training sampling may miss rare cases; six candidates can miss the reference; local projection distorts large extents; decoder constraints can harm some trips; bounded graph search can force breaks; no reserved-evaluator generalisation claim can be made.')
    para('Sources: supplied PathSnap Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, and Assessment and Plagiarism Policy. Technical references: scikit-learn GroupKFold documentation; Optuna TPESampler documentation. See README for URLs.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#687a8d'));canvas.drawString(42,25,'PathSnap | recorded development experiments | team review required');canvas.drawRightString(553,25,str(doc.page))
    doc=SimpleDocTemplate(str(out/'PathSnap_Technical_Report.pdf'),pagesize=(595,842),rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=45)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out/'PathSnap_Technical_Report.pdf')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);a=p.parse_args();build(a.data_root)
