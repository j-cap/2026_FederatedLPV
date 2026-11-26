"""Build an editable draw.io overview and a vector layout preview.

The preview uses the same geometry with Matplotlib mathtext. It is not a native
draw.io PDF export; MathJax/Times New Roman rendering in draw.io may differ.
Running this explicitly rebuilds the new diagram, not the historical overview.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
WIDTH,HEIGHT=1040,660


def build(output):
    output.mkdir(parents=True,exist_ok=True)
    mx=ET.Element('mxfile',host='app.diagrams.net',compressed='false',type='device')
    diagram=ET.SubElement(mx,'diagram',id='personalized-physical-id',name='Personalized physical identification')
    model=ET.SubElement(diagram,'mxGraphModel',dx=str(WIDTH),dy=str(HEIGHT),grid='1',gridSize='10',guides='1',
        tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(WIDTH),pageHeight=str(HEIGHT),math='1',background='#FFFFFF')
    root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
    fig=plt.figure(figsize=(7.16,7.16*HEIGHT/WIDTH));ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,WIDTH),ylim=(HEIGHT,0));ax.axis('off')
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif',
                        'pdf.fonttype':42,'svg.fonttype':'path','svg.hashsalt':'personalized-id-overview'})
    scale=7.16*72/WIDTH;bounds={};texts=[]
    def cell(id,value,style,parent='1',vertex=True):
        return ET.SubElement(root,'mxCell',id=id,value=value,style=style,parent=parent,**({'vertex':'1'} if vertex else {'edge':'1'}))
    def group(id,x,y,w,h,fill='#f7f9fb',stroke='#bdc7ce',parent='1'):
        bounds[id]=(x,y,w,h)
        px,py=bounds[parent][:2] if parent!='1' else (0,0)
        c=cell(id,'','group;connectable=1;',parent);ET.SubElement(c,'mxGeometry',x=str(x-px),y=str(y-py),width=str(w),height=str(h),**{'as':'geometry'})
        c=cell(id+'_frame','',f'rounded=0;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1;',id)
        ET.SubElement(c,'mxGeometry',width=str(w),height=str(h),**{'as':'geometry'})
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=stroke,linewidth=.6))
    def text(id,x,y,w,h,value,size=17,bold=False,math=False,parent='1',color='#172b38'):
        px,py= bounds[parent][:2] if parent!='1' else (0,0)
        style=f'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontFamily=Times New Roman;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};'
        c=cell(id,'$$'+value+'$$' if math else value.replace('\n','<br>'),style,parent)
        ET.SubElement(c,'mxGeometry',x=str(x-px),y=str(y-py),width=str(w),height=str(h),**{'as':'geometry'})
        artist=ax.text(x+w/2,y+h/2,'$'+value+'$' if math else value,ha='center',va='center',fontsize=size*scale,
                       fontweight='bold' if bold else 'normal',color=color,linespacing=1.25)
        texts.append((id,artist,x,y,w,h))
    def arrow(id,start,end,source=None,target=None,points=None,color='#536572'):
        style=f'edgeStyle=none;rounded=0;html=1;strokeColor={color};strokeWidth=1.4;endArrow=block;endFill=1;endSize=7;'
        if source:
            x,y,w,h=bounds[source];style+=f'exitX={(start[0]-x)/w};exitY={(start[1]-y)/h};exitPerimeter=0;'
        if target:
            x,y,w,h=bounds[target];style+=f'entryX={(end[0]-x)/w};entryY={(end[1]-y)/h};entryPerimeter=0;'
        c=cell(id,'',style,vertex=False)
        if source:c.set('source',source)
        if target:c.set('target',target)
        geom=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
        ET.SubElement(geom,'mxPoint',x=str(start[0]),y=str(start[1]),**{'as':'sourcePoint'})
        ET.SubElement(geom,'mxPoint',x=str(end[0]),y=str(end[1]),**{'as':'targetPoint'})
        route=[start]+(points or [])+[end]
        if points:
            a=ET.SubElement(geom,'Array',**{'as':'points'})
            for x,y in points:ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
        for a,b in zip(route[:-2],route[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,linewidth=.8)
        ax.add_patch(FancyArrowPatch(route[-2],route[-1],arrowstyle='-|>',mutation_scale=7,linewidth=.8,color=color,shrinkA=0,shrinkB=0))

    group('local',20,30,655,320,fill='#ffffff',stroke='#aebdc8')
    text('local_title',40,42,615,32,r'On each vehicle $i$',20,True,parent='local')
    # Math is a separate editable label, rather than mixed TeX in ordinary text.
    next(c for c in root if c.get('id')=='local_title').set('value','On each vehicle <i>i</i>')
    group('record',40,100,250,210,parent='local')
    text('record_title',50,112,230,26,'Local measurements',19,True,parent='record')
    text('record_duration',50,150,230,44,'One 12 s record\nat one speed',17,parent='record')
    text('record_speed',50,199,230,30,r'v_i\in\{13,21,29\}\ \mathrm{m/s}',17,math=True,parent='record')
    text('record_known',50,239,230,24,'Known steering and speed',16,parent='record')
    text('record_yaw',50,269,230,30,r'y_{i,k}=r_{i,k}+\epsilon_{i,k}',18,math=True,parent='record')
    group('update',335,100,315,210,fill='#f5f8fa',parent='local')
    text('update_title',345,112,295,28,'Personalized local update',19,True,parent='update')
    text('update_logs',345,151,295,31,r'\mathbf{z}_i=\log\boldsymbol{\theta}_i\in\mathbb{R}^{6}',18,math=True,parent='update')
    text('update_loss',345,191,295,35,r'\min_{\mathbf{z}_i}\ \{L_i(\mathbf{z}_i)',19,math=True,parent='update')
    text('update_prior',345,227,295,36,r'+\ \|\mathbf{W}(\mathbf{z}_i-\boldsymbol{\eta})\|_2^2\}',19,math=True,parent='update')
    text('update_model',345,276,295,23,'Four-state physical model',17,parent='update')
    text('local_footer',40,318,400,24,'Trajectories stay on the vehicle',17,parent='local')
    group('server',745,100,275,210,fill='#eaf5f3',stroke='#64a398')
    text('server_title',755,112,255,28,'Learned fleet prior',20,True,parent='server')
    text('server_mean',755,148,255,77,r'\boldsymbol{\eta}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{z}_j',19,math=True,parent='server')
    text('server_desc',755,230,255,24,'Mean log parameters',17,parent='server')
    text('server_personal',755,271,255,24,'One prior; M local models',17,parent='server')
    arrow('data_to_fit',(290,206),(335,206),'record','update')
    arrow('upload',(650,180),(745,180),'update','server')
    text('upload_label',658,146,79,28,r'\mathbf{z}_i,\ L_i',17,math=True)
    arrow('download',(745,251),(650,251),'server','update',color='#087d87')
    text('download_label',658,216,79,28,r'\boldsymbol{\eta}',19,math=True,color='#087d87')

    group('coverage',20,405,655,210,fill='#f8fafb')
    text('coverage_title',35,416,625,28,'Matched peer-speed comparison',20,True,parent='coverage')
    text('coverage_target',35,450,290,24,'Example: target at 13 m/s',16,parent='coverage')
    for j,v in enumerate([13,21,29]):text(f'speed_{v}',330+100*j,471,90,24,f'{v} m/s',17,True,parent='coverage')
    for row,(name,label,values,color) in enumerate([
        ('same','Same-speed peers',[8,0,0],'#8061a8'),('complementary','Complementary peers',[2,3,3],'#087d87')]):
        y=506+37*row
        text(name+'_label',35,y,285,25,label,18,parent='coverage',color=color)
        for j,value in enumerate(values):text(f'{name}_{j}',330+100*j,y,90,25,str(value),19,True,parent='coverage',color=color)
    text('coverage_footer',35,585,625,23,'Same target record, 8 peer identities and total duration',16,parent='coverage')
    group('evaluation',745,405,275,210,fill='#ffffff',stroke='#64a398')
    text('evaluation_title',755,417,255,28,'Unseen-speed prediction',18,True,parent='evaluation')
    text('evaluation_speed',755,453,255,31,r'v\ne v_i',21,math=True,parent='evaluation')
    text('evaluation_fixed',755,493,255,25,'Fixed learned parameters',17,parent='evaluation')
    text('evaluation_response',755,527,255,36,r'\hat r(\boldsymbol{\theta}_i,\delta,v)',20,math=True,parent='evaluation')
    text('evaluation_score',755,579,255,24,'Clean yaw truth for scoring',16,parent='evaluation')
    arrow('model_to_test',(492.5,310),(882.5,405),'update','evaluation',points=[(492.5,375),(882.5,375)])
    text('model_label',530,343,290,28,r'\boldsymbol{\theta}_i=\exp(\mathbf{z}_i)',18,math=True)
    text('baseline_note',20,629,1000,24,'Reference baseline: the same local objective with a fixed nominal prior',17)

    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for id,artist,x,y,w,h in texts:
        box=artist.get_window_extent(renderer)
        p0=ax.transData.transform((x,y));p1=ax.transData.transform((x+w,y+h))
        if box.width > abs(p1[0]-p0[0])+1 or box.height > abs(p1[1]-p0[1])+1:
            raise ValueError(f'Preview label does not fit: {id}, text {box.width:.1f} x {box.height:.1f}, cell {abs(p1[0]-p0[0]):.1f} x {abs(p1[1]-p0[1]):.1f}')
    ET.indent(mx,space='  ')
    path=output/'personalized_id_overview.drawio'
    ET.ElementTree(mx).write(path,encoding='utf-8',xml_declaration=True)
    for suffix in ['svg','pdf','png']:
        metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else {'Date':None} if suffix=='svg' else None
        preview=output/f'personalized_id_overview_preview.{suffix}'
        fig.savefig(preview,dpi=180,metadata=metadata)
        if suffix=='svg':
            preview.write_text('\n'.join(line.rstrip() for line in preview.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
    print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'paper/figures/personalized_id')
    build(parser.parse_args().output_dir.resolve())
