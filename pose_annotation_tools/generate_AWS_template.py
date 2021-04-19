import argparse
import yaml
import os, sys

# create template for an AWS labeling job
def generate_AWS_template(config):

	with open(config) as f:
		data = yaml.load(f)
	data['labels'] = [n + " " + data['species'] + " " + k for n in data['animal_names'] for k in data['keypoints']]

	project_dir = os.path.basename(config)
	f = open(os.path.join(project_dir,'annotation_interface.template'),'w')

	message = """<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

	<crowd-form>
	  <div id="errorBox">
	  </div>
		<crowd-keypoint
			src="{{{{ task.input.taskObject | grant_read_access }}}}"
			labels="{labels}"
			header="{header}"
			name="annotatedResult">

			  <short-instructions>
			  {short_instructions}
			  </short-instructions>

			  <full-instructions header="Instructions" >
			  {full_instructions}
			  </full-instructions>

		</crowd-keypoint>
	<br></br>
	<!--Additional instructions/sample images could go here-->

	</crowd-form>


	<!----------------------------------------Script to ensure each body part is annotated exactly N times------------------------------------------------>
	<script>
	  var num_obj = {num_obj}; // if you are tracking more than one animal (and don't have separate keypoint names for each animal.)

	  document.querySelector('crowd-form').onsubmit = function(e) {{
		const keypoints = document.querySelector('crowd-keypoint').value.keypoints || document.querySelector('crowd-keypoint')._submittableValue.keypoints;
		const labels = keypoints.map(function(p) {{
		return p.label;
		}});

		// 1. Make sure total number of keypoints is correct.
		var original_num_labels = document.getElementsByTagName("crowd-keypoint")[0].getAttribute("labels");

		original_num_labels = original_num_labels.substring(2, original_num_labels.length - 2).split("\",\"");
		var goalNumKeypoints = num_obj*original_num_labels.length;
		if (keypoints.length != goalNumKeypoints) {{
		  e.preventDefault();
		  errorBox.innerHTML = '<crowd-alert type="error">You must add all keypoint annotations and use each label only once.</crowd-alert>';
		  errorBox.scrollIntoView();
		  return;
		}}

		// 2. Make sure all labels are unique.
		labelCounts = {{}};
		for (var i = 0; i < labels.length; i++) {{
		  if(!labelCounts[labels[i]]) {{
			labelCounts[labels[i]] = 0;
		  }}
		  labelCounts[labels[i]]++;
		}}
		const goalNumSingleLabel = num_obj;

		const numLabels = Object.keys(labelCounts).length;

		Object.entries(labelCounts).forEach(entry => {{
		  if (entry[1] != goalNumSingleLabel) {{
			e.preventDefault();
			errorBox.innerHTML = '<crowd-alert type="error">You must use each label only once.</crowd-alert>';
			errorBox.scrollIntoView();
		  }}
		}})
	  }};
	</script>

	<style>
	img {{
	  display: block;
	  margin-left: auto;
	  margin-right: auto;
	}}
	</style>
	""".format(**data)

	f.write(message)
	f.close()


if __name__ ==  '__main__':
    """
    generate_AWS_template command line entry point
    Arguments:
        config 		Path to project_config.yml for your MARS annotation project
    """

    parser = argparse.ArgumentParser(description='generate annotation interface html template from project_config.yml', prog='generate_AWS_template')
    parser.add_argument('config', type=str, help="Path to project_config.yml for your annotation project")
	args = parser.parse_args(sys.argv[1:])

    generate_AWS_template(args.config)
