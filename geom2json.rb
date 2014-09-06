# JSON Exporter for Google Sketchup
# Jukka Saarelma  2014
# Aalto University
#
# Exporter is heavily influenced by 
# VTK Exporter for Google Sketchup
# By Alex Southern
#


require 'sketchup.rb'

def jsonExport
  mod = Sketchup.active_model
  ent = mod.entities
  sel = mod.selection
  layers = mod.layers
  $num_l = layers.length

  model_filename = File.basename(mod.path)
    if sel.empty?
      ans = UI.messagebox("No objects selected. Export entire model?", MB_YESNOCANCEL)
      if( ans == 6 )
        $model = mod.entities
      else
        $model = sel
      end
    else
      $model = Sketchup.active_model.selection
    end
    
  $vrtx = []
  $vrtx_len = 0

  $numPolys = 0
  $numUsedPts = 0
  
  if ($model.length > 0)
     file_type="json"
              
     out_name = UI.savepanel( file_type+" file location", "." , "#{File.basename(mod.path).split(".")[0]}." +file_type )
     
     $mesh_file = File.new( out_name , "w" )  
     model_name = model_filename.split(".")[0]
     $mesh_file.puts("{")
     jsonExtractData
     jsonWritePoints
     jsonWritePolygons
     jsonWriteLayers
     $mesh_file.puts("}") 
     $mesh_file.close
    
  end
end # End main
  
def jsonExtractData
   $numPolys = 0
   $numUsedPts = 0
   $model.each do |entity|
      if (entity.typename == "Face")
        # Each Face is made up of triangular polygons, keep a count of how many polygons
        $numPolys += entity.mesh.count_polygons
        entity.vertices.each do |vertex| 
          $numUsedPts += 1
          if ($vrtx_len == 0)
             # Add First Vertex to Vrtx List
             $vrtx[$vrtx_len] = vertex.position.to_a
             $vrtx_len += 1             
          else
             # Check if current vertex already exists in $vrtx list
             found = 0 
             (0..$vrtx_len).each { |i|
               if ($vrtx[i] == vertex.position.to_a)
                 found = 1
               end
             }
             # If vertex is unique then add it to the $vrtx list       
             if (found == 0)
               $vrtx[$vrtx_len] = vertex.position.to_a
               $vrtx_len += 1             
             end     
          end           
        end
               
      end
   end  
end # End function
 
def jsonWritePoints
   $mesh_file.puts("\t\"vertices\": [")
    (0..$vrtx_len-1).each { |i|
      $mesh_file.print("\t")
      (0..$vrtx[i].length-1).each {|j| 
        v = $vrtx[i][j]*0.0254

        $mesh_file.print(v)
        if( (i != $vrtx_len-1) or (j != 2))
          $mesh_file.print(",")
        end
      }

    $mesh_file.puts(" ")
   }
   $mesh_file.puts("\t],")
end # End function

def jsonWritePolygons
  # Calculate number of triangles
  tri_count = 0
  count = 0
  $mesh_file.puts("\t\"indices\": [")
  $model.each do |entity|
    if (entity.typename == "Face")
      mesh = entity.mesh
      num = mesh.count_polygons
      (1..num).each { |i|
        count += 1
        $mesh_file.print("\t")
        p = mesh.polygon_at(i)
        p1 = mesh.point_at(p[0].abs)
        p2 = mesh.point_at(p[1].abs)  
        p3 = mesh.point_at(p[2].abs)
        i1 = $vrtx.index(p1.to_a)
        i2 = $vrtx.index(p2.to_a) 
        i3 = $vrtx.index(p3.to_a)
        str = i1.to_s+", "+i2.to_s+", "+i3.to_s  
        $mesh_file.print(str)
        if(count != $numPolys)
          $mesh_file.puts(",")
        end
      }
    end # Face if
   end # Entity loop  
   $mesh_file.puts("\n\t],")
end # End function
      
   
def jsonWriteLayers
  m = $model.model;
  l = []
  l_num = 0;
  flag = 0;
    
  $mesh_file.puts("\t\"layers_of_triangles\": [")
  count = 0
  $model.each do |entity|
  if(entity.typename == "Face")
    flag = 0
    layername = entity.layer.name
       
    (0..l_num).each {|i|
      if (l[i] == layername && l_num >0)
        flag = 1
      end
    }
       
    if (flag == 0)
      l_num+=1
      l[l_num-1] = layername
    end
       
    mesh = entity.mesh
    mesh.polygons.each do |poly|
      count += 1
      $mesh_file.print("\t\"")
      $mesh_file.print(layername)
      $mesh_file.print("\"")
      if(count < $numPolys)
        $mesh_file.print(",")
      end
      $mesh_file.print("\n")
    end
  end # End face if
  end # End entity for

  $mesh_file.puts("\t],")

  $mesh_file.puts("\t\"layer_names\": [")
  (0..l_num-1).each{|i|
    $mesh_file.print("\t\"")
    $mesh_file.print(l[i].to_s)
    $mesh_file.print("\"")
    if(i < l_num-1)
      $mesh_file.print(",")
    end
    $mesh_file.print("\n")
  }
  $mesh_file.puts("\t]")
end # End function

if (not file_loaded?("geom2json.rb"))
  UI.menu("Tools").add_item("Export to JSON") {jsonExport}
end

file_loaded("geom2json.rb")


