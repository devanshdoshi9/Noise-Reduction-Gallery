<!DOCTYPE html>
<html lang="en">
    <head>
        
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <!--
        <link rel="stylesheet"  type="text/css" href="resources/css/style.css"> 
        -->
        <link rel="stylesheet" href="css/imgareaselect.css">
        <title>Gallery</title>
    </head>

    <body>
        <form action="crop.php" method="post" enctype="multipart/form-data">
            Upload Image: <input type="file" name="image" id="image" />
            <input type="hidden" name="x1" value="" />
            <input type="hidden" name="y1" value="" />
            <input type="hidden" name="w" value="" />
            <input type="hidden" name="h" value="" /><br><br>
            <input type="submit" name="submit" value="Submit" />
        </form>
 
        <p><img id="previewimage" style="display:none;"/></p>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
        <script src="js/jquery.imgareaselect.min.js"></script>
        
        <script>
            jQuery(function($) {

                var p = $("#previewimage");
                $("body").on("change", "#image", function(){

                    var imageReader = new FileReader();
                    imageReader.readAsDataURL(document.getElementById("image").files[0]);

                    imageReader.onload = function (oFREvent) {
                        p.attr('src', oFREvent.target.result).fadeIn();
                    };
                });

                $('#previewimage').imgAreaSelect({
                    onSelectEnd: function (img, selection) {
                        $('input[name="x1"]').val(selection.x1);
                        $('input[name="y1"]').val(selection.y1);
                        $('input[name="w"]').val(selection.width);
                        $('input[name="h"]').val(selection.height);            
                    }
                });
            });
        </script>
    </body>