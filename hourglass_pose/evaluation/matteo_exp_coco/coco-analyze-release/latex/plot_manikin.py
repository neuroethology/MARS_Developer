import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np


I=io.imread('coco-analyze-release/latex/lab_mouse_black.png')
xy =np.array([[175,282,164,223,256,138,140],
             [30,97,138,122,297,226,384]])
colors=['b','r','g','orange','r','g','c']
ec = ['#cd87ff','#cd87ff','#cd87ff','#74c8f9','#74c8f9','#feff95','#feff95']

fig = plt.figure(figsize=(5,15))
plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99)
plt.axis('off')
plt.imshow(I)

plt.plot([xy[0,0],xy[0,1]],[xy[1,0],xy[1,1]],color='#cd87ff',lw=3)
plt.plot([xy[0,0],xy[0,2]],[xy[1,0],xy[1,2]],color='#cd87ff',lw=3)
plt.plot([xy[0,1],xy[0,2]],[xy[1,1],xy[1,2]],color='#cd87ff',lw=3)
plt.plot([xy[0,3],xy[0,4]],[xy[1,3],xy[1,4]],color='#74c8f9',lw=3)
plt.plot([xy[0,3],xy[0,5]],[xy[1,3],xy[1,5]],color='#74c8f9',lw=3)
plt.plot([xy[0,4],xy[0,6]],[xy[1,4],xy[1,6]],color='#feff95',lw=3)
plt.plot([xy[0,5],xy[0,6]],[xy[1,5],xy[1,6]],color='#feff95',lw=3)

for i in range(7): plt.plot(xy[0,i],xy[1,i],'o',markersize=15,color=colors[i],
                            markeredgecolor=ec[i],markeredgewidth=3)
plt.tight_layout()
plt.savefig('coco-analyze-release/latex/manikin_black.png',transparent=True)
plt.savefig('coco-analyze-release/latex/manikin_black.pdf',transparent=True)


I=io.imread('coco-analyze-release/latex/lab_mouse_white.png')
xy =np.array([[175,282,164,223,256,138,140],[30,97,138,122,297,226,384]])
colors=['#21a5f2','#ed42b9','#46db41','#ef571a','#ed42b9','#46db41','#35fbff']
ec = ['#ae16f9','#ae16f9','#ae16f9','#21a5f2','#21a5f2','#f99c4a','#f99c4a']

fig = plt.figure(figsize=(5,15))
plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99)
plt.axis('off')
plt.imshow(I)

plt.plot([xy[0,0],xy[0,1]],[xy[1,0],xy[1,1]],color='#ae16f9',lw=3)
plt.plot([xy[0,0],xy[0,2]],[xy[1,0],xy[1,2]],color='#ae16f9',lw=3)
plt.plot([xy[0,1],xy[0,2]],[xy[1,1],xy[1,2]],color='#ae16f9',lw=3)
plt.plot([xy[0,3],xy[0,4]],[xy[1,3],xy[1,4]],color='#21a5f2',lw=3)
plt.plot([xy[0,3],xy[0,5]],[xy[1,3],xy[1,5]],color='#21a5f2',lw=3)
plt.plot([xy[0,4],xy[0,6]],[xy[1,4],xy[1,6]],color='#f99c4a',lw=3)
plt.plot([xy[0,5],xy[0,6]],[xy[1,5],xy[1,6]],color='#f99c4a',lw=3)

for i in range(7): plt.plot(xy[0,i],xy[1,i],'o',markersize=15,color=colors[i],
                            markeredgecolor=ec[i],markeredgewidth=3)
plt.tight_layout()
plt.savefig('coco-analyze-release/latex/manikin_white.png',transparent=True)
plt.savefig('coco-analyze-release/latex/manikin_white.pdf',transparent=True)


# fig = plt.figure(figsize=(5,10))
# plt.subplots_adjust(hspace=0.5)
# plt.subplot(3,1,1)
# plt.plot([xy[0,0],xy[0,1]],[xy[1,0],xy[1,1]],color='#cd87ff',lw=3)
# plt.plot([xy[0,0],xy[0,2]],[xy[1,0],xy[1,2]],color='#cd87ff',lw=3)
# plt.plot([xy[0,1],xy[0,2]],[xy[1,1],xy[1,2]],color='#cd87ff',lw=3)
# for i in range(3): plt.plot(xy[0,i],xy[1,i],'o',markersize=15,color=colors[i],
#                             markeredgecolor=ec[i],markeredgewidth=3,clip_on=False)
# plt.axis('equal')
# plt.axis('off')
# plt.gca().invert_yaxis()
# plt.title('Head')
#
# plt.subplot(3,1,2)
# plt.plot([xy[0,3],xy[0,4]],[xy[1,3],xy[1,4]],color='#74c8f9',lw=3)
# plt.plot([xy[0,3],xy[0,5]],[xy[1,3],xy[1,5]],color='#74c8f9',lw=3)
# for i in range(3,6): plt.plot(xy[0,i],xy[1,i],'o',markersize=15,color=colors[i],
#                             markeredgecolor=ec[i],markeredgewidth=3,clip_on=False)
# plt.axis('equal')
# plt.axis('off')
# plt.gca().invert_yaxis()
# plt.title('Upper body')
#
# plt.subplot(3,1,3)
# plt.plot([xy[0,4],xy[0,6]],[xy[1,4],xy[1,6]],color='#feff95',lw=3)
# plt.plot([xy[0,5],xy[0,6]],[xy[1,5],xy[1,6]],color='#feff95',lw=3)
# for i in range(4,7): plt.plot(xy[0,i],xy[1,i],'o',markersize=15,color=colors[i],
#                             markeredgecolor=ec[i],markeredgewidth=3,clip_on=False)
# plt.axis('equal')
# plt.axis('off')
# plt.gca().invert_yaxis()
# plt.title('Lower body')
# plt.tight_layout()
# plt.savefig('coco-analyze-release/latex/minikin_parts.png')
# plt.savefig('coco-analyze-release/latex/minikin_parts.pdf')